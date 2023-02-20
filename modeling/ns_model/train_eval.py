# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json
import os
import numpy as np
import torch
import shutil
import random
from dotmap import DotMap
import argparse
import copy
from tqdm import tqdm 
import pickle

from modeling.ns_model.model.learned import LearnedModel
from modeling.ns_model.utils import model_util, utils, data_util, config
from modeling.ns_model.ns_data_generator import ArenaNSDataset

def prepare(args):
    '''
    create logdirs, check dataset, seed pseudo-random generators
    '''
    
    # set seeds
    torch.manual_seed(args.seed)
    random.seed(a=args.seed)
    np.random.seed(args.seed)
    args.dout = os.path.join(args.checkpt_dir, str(args.exp_num))
    # make output dir
    print(args)
    if not os.path.isdir(args.dout):
        os.makedirs(args.dout)
    return args


def create_model(args, embs_ann, vocab_in, vocab_out):
    '''
    load a model and its optimizer
    '''
    prev_train_info = model_util.load_log(args.dout, stage='train')
    if args.resume and os.path.exists(os.path.join(args.dout, 'latest.pth')):
        # load a saved model
        loadpath = os.path.join(args.dout, 'latest.pth')
        model, optimizer = model_util.load_model(
            loadpath, args.device, prev_train_info['progress'] - 1)
        assert model.vocab_out.contains_same_content(vocab_out)
        model.args = args
    else:
        # create a new model
        if not args.resume and os.path.isdir(args.dout):
            shutil.rmtree(args.dout)
        model = LearnedModel(args, embs_ann, vocab_in, vocab_out)
        model = model.to(torch.device(args.device))
        optimizer = None
            
    # put encoder on several GPUs if asked
    if torch.cuda.device_count() > 1:
        print('Parallelizing the model')
        model.model = utils.DataParallel(model.model)
    return model, optimizer, prev_train_info


def load_data(args, split, ann_type):
    dataset = ArenaNSDataset(
        args=args,
        metadata_file=args.data_dir + "trajectory-data/%s.json" % split,
        images_root=args.data_dir + "trajectory-data/mission_images",
        split=split,
        annotation_type=ann_type)

    return dataset


def process_vocabs(datasets, args):
    '''
    assign the largest output vocab to all datasets, compute embedding sizes
    '''
    # find the longest vocabulary for outputs among all datasets
    vocab_out = sorted(datasets, key=lambda x: len(x.vocab_out))[-1].vocab_out
    vocab_in = sorted(datasets, key=lambda x: len(x.vocab_in))[-1].vocab_in
    # make all datasets to use this vocabulary for outputs translation
    for dataset in datasets:
        dataset.vocab_translate = vocab_out
        dataset.save_vocab_in()
        # dataset.vocab_in = vocab_in
    # prepare a dictionary for embeddings initialization: vocab names and their sizes
    embs_ann = {}
    for dataset in datasets:
        embs_ann[dataset.split + ":" + dataset.name] = len(dataset.vocab_in)
    return embs_ann, vocab_in, vocab_out


def wrap_datasets(datasets, args):
    '''
    wrap datasets with torch loaders
    '''
    loaders = {}
    loader_args = {
        'num_workers': args.num_workers,
        'drop_last': (torch.cuda.device_count() > 1)}
    for dataset in datasets:
        if 'train' in dataset.split:
            weights = [1 / len(dataset)] * len(dataset)
            num_samples = len(dataset)
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, num_samples=num_samples, replacement=True)
            loader = torch.utils.data.DataLoader(
                dataset, args.batch_size, sampler=sampler, collate_fn=utils.collate_fn, **loader_args)
        elif 'valid' in dataset.split:
            loader = torch.utils.data.DataLoader(
                dataset, args.batch_size, shuffle=False, collate_fn=utils.collate_fn, **loader_args)

        loaders[dataset.split + ":" + dataset.name] = loader
        
    return loaders


def train_model(args, eval_split):
    '''
    train a network using an lmdb dataset
    '''
    print("Start training ------------------------------")
    args = prepare(args)
    # load dataset(s) and process vocabs
    datasets = []
    datasets.append(load_data(args, eval_split, ann_type=args.ann_type))
    
    # # assign vocabs to datasets and check their sizes for nn.Embeding inits
    embs_ann, vocab_in, vocab_out = process_vocabs(datasets, args)
    # # wrap datasets with loaders
    loaders = wrap_datasets(datasets, args)
    # # create the model
    model, optimizer, prev_train_info = create_model(args, embs_ann, vocab_in, vocab_out)
    # # start train loop
    model.run_train(loaders, prev_train_info, optimizer=optimizer)


def eval_model(args, eval_split):
    print("Start evaluation ------------------------------")
    model_path = args.checkpt_dir+str(args.exp_num)+"/"
    with open(model_path+"config.json", "r") as f:
        args_saved = json.load(f)
    
    args = DotMap(args_saved)
    args.debug = False
    # load the dataset to evaluate
    dataset = load_data(args, eval_split, ann_type=str(args.ann_type))

    # load the training vocab
    with open(model_path+"%s_%s_vocabin.pkl" % (str(args.ann_type), "train"), "rb") as f:
        vocab_in = pickle.load(f)
    
    learned_model, _ = model_util.load_model(model_path+"latest.pth", args.device)
    model = learned_model.model
    model.eval()
    model.args.device = args.device

    vocab = {'word': vocab_in, 'action_low': model.vocab_out}
    action_suc_cnt = 0
    annot_suc_cnt = 0
    lang_suc_cnt = 0
    total_actions = 0
    total_annot = 0
    mission_suc = {}

    for d_full in tqdm(dataset):
        total_annot += 1
        mission_id = d_full["mission_id"]
        if mission_id not in mission_suc:
            mission_suc[mission_id] = []
        prev_action = None
        action_idx = d_full["action_idx"]
        lang_idx = []
        for l_i in range(len(action_idx)):
            a_num = action_idx[l_i]
            lang_idx += [l_i for l_cnt in a_num]
        
        model.reset()
        annot_succ = []
        for t in range(len(d_full["action"])):
            d_t = copy.deepcopy(d_full)
            d_t["action"] = [d_full['action'][t:t+1]]
            d_t["action_valid_interact"] = [d_full['action_valid_interact'][t:t+1]]
            d_t["object"] = [d_full['object'][t:t+1]]
            d_t["frames"] = [d_full['frames'][t:t+1]]
            d_t["lang"] = [d_full["lang"]]

            gt_action = d_t["action"][0][0]
            gt_object = d_t["object"][0][0]
            input_dict, gt_dict = data_util.tensorize_and_pad(d_t, vocab_in, model.vocab_out, args.device, model.pad)
            with torch.no_grad():
                m_out = model.step(input_dict, vocab, prev_action=prev_action)
            m_pred = model_util.extract_action_preds(m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
            action_pred = m_pred['action']
            obj_pred = model.vocab_out.index2word[m_pred['object'][0][0]]
            if args.debug:
                print("Language: %s" % d_t["lang"][0])
                print("GT action and object: %s %s" % (gt_action, gt_object))
                print("Predicted action and object: %s %s" % (action_pred, obj_pred))

            if action_pred == gt_action and obj_pred == gt_object:
                action_suc_cnt += 1
                annot_succ.append(True)
            else:
                annot_succ.append(False)

            total_actions += 1
            prev_action = str(action_pred)
        
        annot_suc_array = np.array(annot_succ)
        if annot_suc_array.all():
            annot_suc_cnt += 1
        mission_suc[mission_id].append(annot_succ)

    action_acc = action_suc_cnt/total_actions
    annot_acc = annot_suc_cnt/total_annot
    with open(model_path+"mission_suc.json", "w") as f:
        json.dump(mission_suc, f, indent=4)

    print("action level accuracy: %s" % str(action_acc))
    print("mission level accuracy: %s" % str(annot_acc))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", dest="device", type=int,
                        default=0, help="The device id to run the model.")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    parser.add_argument("--exp-num", dest="exp_num", type=int, default=2,
                        help="exp id for tracking purposes")
    parser.add_argument("--debug", dest="debug", type=bool, default=False)
    parser.add_argument("--model-dir", dest="model_dir", type=str, default=os.getenv('ALEXA_ARENA_DIR')+"/modeling/")
    parser.add_argument("--vision-model-path", dest="vision_model_path", type=str, default=os.getenv('ALEXA_ARENA_DIR') + '/logs/vision_model_checkpt/21.pth')
    parser.add_argument("--checkpt-dir", dest="checkpt_dir", type=str, default=os.getenv('ALEXA_ARENA_DIR') + "/logs/ns_model_checkpt/")
    parser.add_argument("--data-dir", dest="data_dir", type=str, default=os.getenv('ALEXA_ARENA_DIR')+"/data/")
    parser.add_argument("--ann-type", dest="ann_type", type=str, default="h")
    parser.add_argument("--name", dest="name", type=str, default="experiment 1")
    parser.add_argument("--resume", dest="resume", type=bool, default=True)
    parser.add_argument("--seed", dest="seed", type=int, default=12345)
    parser.add_argument("--model", dest="model", type=str, default="transformer")
    parser.add_argument("--dropout", dest="dropout", type=dict, default=config.dropout)
    parser.add_argument("--lr", dest="lr", type=dict, default=config.lr)
    parser.add_argument("--encoder_lang", dest="encoder_lang", type=dict, default=config.encoder_lang)
    parser.add_argument("--decoder_lang", dest="decoder_lang", type=dict, default=config.decoder_lang)
    parser.add_argument("--enc", dest="enc", type=dict, default=config.enc)
    parser.add_argument("--detach_lang_emb", dest="detach_lang_emb", type=bool, default=config.detach_lang_emb)
    parser.add_argument("--encoder_heads", dest="encoder_heads", type=int, default=config.encoder_heads)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=config.batch_size)
    parser.add_argument("--num-epochs", dest="epochs", type=int, default=config.epochs)
    parser.add_argument("--encoder_layers", dest="encoder_layers", type=int, default=config.encoder_layers)
    parser.add_argument("--num_input_actions", dest="num_input_actions", type=int, default=config.num_input_actions)
    parser.add_argument("--demb", dest="demb", type=int, default=config.demb)
    parser.add_argument("--weight_decay", dest="weight_decay", type=int, default=config.weight_decay)
    parser.add_argument("--action_loss_wt", dest="action_loss_wt", type=float, default=config.action_loss_wt)
    parser.add_argument("--object_loss_wt", dest="object_loss_wt", type=float, default=config.object_loss_wt)
    parser.add_argument("--optimizer", dest="optimizer", type=str, default=config.optimizer)
    parser.add_argument("--visual_archi", dest="visual_archi", type=str, default=config.visual_archi)

    args = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    train_model(args, "train")
    eval_model(args, "valid")