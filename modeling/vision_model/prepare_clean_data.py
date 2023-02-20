# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os
from glob import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from data_cleaning_version_control import collapse_dict_versions, data_is_valid_v5, decode_absent_class_assignments
from collections import defaultdict
from functools import partial
from argparse import ArgumentParser
import constants
from p_tqdm import p_map


def process(data_is_valid, object_id_to_class, class_to_area_thresholds, classes_list, split, met_path):
    class_counter = defaultdict(int)
    unfound_classes, pruned_files = set(), []
    try:
        with open(met_path) as f1:
            metadata = json.load(f1)
    except json.decoder.JSONDecodeError:
        return unfound_classes, pruned_files, class_counter 
    # Add only valid images
    if data_is_valid(metadata["image_annotations"], object_id_to_class, class_to_area_thresholds, classes_list):
        annotations = metadata["image_annotations"]
        for ann in annotations:
            object_id_long = ann["object_id"]
            object_id = '_'.join(object_id_long.split("_")[:-1])
            bbox = ann['bbox']
            area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            rgb = ann["rgb"]
            if object_id in object_id_to_class:
                class_assignment = object_id_to_class[object_id]
            else:
                class_assignment = \
                    decode_absent_class_assignments(object_id,
                                                    object_id_long,
                                                    object_id_to_class)
                if class_assignment == None:
                    with open(f"{data_entrypoint_save_dir}/class_assignment_not_found.txt", 'a+') as f:
                        f.write(object_id + '\n')
                    unfound_classes.add(object_id)
            try:
                area_threshold = class_to_area_thresholds[class_assignment][0]
                area_threshold = min(100, area_threshold)
            except:
                area_threshold = 1
            if (class_assignment in classes_list) and area > area_threshold and rgb != [0, 0, 0] and class_assignment != "Unassigned":
                class_counter[class_assignment] += 1
        path_split = met_path.split('/')
        rel_met_path = '/'.join(path_split[-4:])
        with open(f"{data_entrypoint_save_dir}/metadata_{split}.txt", 'a+') as f:
            f.write(rel_met_path + '\n')
        pruned_files.append(met_path)
    return unfound_classes, pruned_files, class_counter

def process_data(args):
    split = args.split
    classes_path = args.classes_path
    if classes_path:
        with open(classes_path) as f:
            classes_list = f.readlines()
    else:
        classes_list = []
    classes_list = set([cl.strip() for cl in classes_list])

    data_root = args.data_root
    data_entrypoint_save_dir = args.data_entrypoint_save_dir
    all_metadata_paths = []
    for data_version_folder in [dr for dr in os.listdir(data_root)]:
        for run_folder in os.listdir(os.path.join(data_root, data_version_folder)):
            path = os.path.join(data_root, data_version_folder, run_folder, split)
            print(path)
            metadata_paths = [y for x in os.walk(path)
                            for y in glob(os.path.join(x[0], '*.json'))]
            print("Number of metadata files:", len(metadata_paths))
            all_metadata_paths.extend(metadata_paths)
    if args.process_small_data_portion:
        all_metadata_paths  = all_metadata_paths[0: 50000]
        print("\nProcessing only part of the data because of --process-small-data-portion flag set. "
              "Unset if not needed.\n")
    print("Number of total metadata files:", len(all_metadata_paths))
    if args.custom_class_to_object_id_path:
        class_to_object_id, object_id_to_class = \
            process_custom_class_object_mappings(args.custom_class_to_object_id_path)
    else:
        class_to_object_id, object_id_to_class =  get_class_object_mapping_from_rg_objects_list(args)
    if len(classes_list) == 0:
        classes_list = set(list(class_to_object_id.keys()))
    else:
        class_to_object_id, object_id_to_class = filter_classes(class_to_object_id, classes_list)

    with open(args.class_to_area_thresholds_path) as f:
        class_to_area_thresholds = json.load(f)
    unfound_classes_all = set()
    data_is_valid = data_is_valid_v5
    pruned_files_all = []
    class_counter_all = defaultdict(int)
    num_processes = 4
    func = partial(process, data_is_valid, object_id_to_class, class_to_area_thresholds, classes_list, split)
    outputs = p_map(func, all_metadata_paths, num_cpus=num_processes)

    for out in outputs:
        unfound_classes_all.update(out[0])
        pruned_files_all.extend(out[1])
        dct = out[2]
        for key in dct:
            class_counter_all[key] += dct[key]
        
    print(f"Number of files after pruning: {len(pruned_files_all)}")
    print("Class frequencies")
    print(class_counter_all)
    with open(f"{data_entrypoint_save_dir}/class_counter_{split}.json", 'w') as f:
        json.dump(class_counter_all, f, indent=4)
    for unk in unfound_classes_all:
        with open(f"data/unfound_classes_{split}.txt", 'a+') as f:
            f.write(unk + '\n')
    class_to_idx = {"Unassigned": 0}
    classes = sorted(list(class_to_object_id.keys()))
    classes.remove("Unassigned")
    class_to_idx.update({clas: i + 1 for i, clas in enumerate(classes)})
    with open(f"{data_entrypoint_save_dir}/class_to_idx.json", "w") as m_file:
        json.dump(class_to_idx, m_file, indent=4)
    print("Number of readable names / classes for the model:", len(class_to_object_id))
    with open(f"{data_entrypoint_save_dir}/class_to_obj_id.json", 'w') as f:
        json.dump(class_to_object_id, f, indent=4)
    with open(f"{data_entrypoint_save_dir}/obj_id_to_class.json", 'w') as f:
        json.dump(object_id_to_class, f, indent=4)
    return all_metadata_paths

def process_custom_class_object_mappings(custom_class_to_object_id_path):
    with open(custom_class_to_object_id_path) as f:
        class_to_object_id = json.load(f)
    # We expect a customly designed already collapsed mappings, so no collapsing here
    object_id_to_class = map_object_id_to_class(class_to_object_id)
    return class_to_object_id, object_id_to_class 

def filter_classes(class_to_object_id, classes_list):
    new_class_to_object_ids = defaultdict(list)
    for clas in class_to_object_id:
        if (clas == "Unassigned") or (clas in classes_list):
            new_class_to_object_ids[clas] = class_to_object_id[clas]
        else:
            new_class_to_object_ids["Unassigned"].extend(class_to_object_id[clas])
    collapse_dict = collapse_dict_versions["v4"]
    class_to_object_id = collapse_classes(class_to_object_id, collapse_dict)
    new_object_id_to_class = map_object_id_to_class(new_class_to_object_ids)
    return new_class_to_object_ids, new_object_id_to_class

def get_class_object_mapping_from_rg_objects_list(args):
    rg_object_list_root = args.rg_object_list_root
    with open(rg_object_list_root) as f:
        all_object_jsons = json.load(f)
    print('Number of objects from RG object list: {}'.format(len(all_object_jsons)))
    class_to_object_id = defaultdict(list)
    for object_desc in all_object_jsons:
        object_id = object_desc["ObjectID"]
        if not object_desc["ReadableName"]:
            if "CanSoda" in object_id:
                readable_name = "Can"
            elif "KitchenCabinet" in object_id:
                readable_name = "Cabinet" 
            elif "Fork_Lift" in object_id:
                readable_name = "Forklift"
            elif  "MissionItemHolder" in object_id:
                readable_name = "Unassigned"
            else:
                readable_name = "Unassigned"
        else:
            readable_name = object_desc["ReadableName"]
        class_to_object_id[readable_name].append(object_id)

    # Something in the metadata and the image segmentation got mapped to Lab Terminal that wasnt found in the object list, so adding here
    class_to_object_id["Machine Panel"].append("Lab_Terminal")
    class_to_object_id["Unassigned"] = ["Unassigned"]
    # Collapse classes
    collapse_dict = collapse_dict_versions["v4"]
    class_to_object_id = collapse_classes(class_to_object_id, collapse_dict)
    object_id_to_class = map_object_id_to_class(class_to_object_id)
    return class_to_object_id, object_id_to_class

def map_object_id_to_class(class_to_object_id):
    object_id_to_class = {}
    for clas in class_to_object_id:
        for object_id in class_to_object_id[clas]:
            object_id_to_class[object_id] = clas
    return object_id_to_class

def collapse_classes(class_to_object_id, collapse_dict):
    """
    collapse_dict = {
        "Table": ["Table", "Desk", "Counter"],
        "Apple": ["Apple", "Apple Slice"]
    }
    means collapse ["Table", "Desk", "Counter"] into class "Table",
    collapse ["Apple", "Apple Slice"] into class "Apple" etc.
    """
    for cluster_class in collapse_dict:
        new_map = []
        for collapse_class in collapse_dict[cluster_class]:
            if collapse_class in class_to_object_id:
                new_map.extend(class_to_object_id[collapse_class])
                # Delete all the classes
                del class_to_object_id[collapse_class]
        class_to_object_id[cluster_class] = list(set(new_map))
    return class_to_object_id
        
            


def plot_class_counter(class_counter_path):
    with open(class_counter_path) as f:
        counter = json.load(f)
    bars = list(counter.keys())
    heights = [counter[cat] for cat in bars]
    zipped_lists = zip(heights, bars)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    heights, bars = [ list(tuple) for tuple in  tuples]
    y_pos = np.arange(len(bars))
    new_x = [2*i for i in y_pos]
    plt.figure(figsize=(20, 8))  # width:10, height:8
    plt.bar(new_x, heights, width=0.8, bottom=None, align='center', data=None)

    # Create names on the x-axis
    plt.xticks(new_x, bars, rotation=90)
    plt.title("Categories vs frequencies in testing data")
    plt.tight_layout()
    
    plt.savefig("class_counter_test.png")
    # Show graphic
    plt.show()

def plot_per_cat_metrics(per_cat_metrics_file_path, class_counter_path):
    with open(class_counter_path) as f:
        counter = json.load(f)
    with open(per_cat_metrics_file_path) as f:
        metrics = json.load(f)
    dir_to_save = "maps"
    if not os.path.exists(dir_to_save):
        os.makedirs(dir_to_save)
    for iou_type in metrics:
        for map_metric in metrics[iou_type].keys():
            freqs = []
            maps = []
            cats = []
            print("Number of categories: ", len(metrics[iou_type][map_metric]))
            for category in metrics[iou_type][map_metric]:
                map_ = metrics[iou_type][map_metric][category]
                freqs.append(counter[category])
                maps.append(map_)
                cats.append(category)
            zipped_lists = zip(freqs, maps, cats)
            sorted_pairs = sorted(zipped_lists)
            tuples = zip(*sorted_pairs)
            freqs, maps, cats = [list(tuple) for tuple in  tuples]
            fig, ax = plt.subplots(figsize=(20,8))
            ax.plot(cats, maps, color='g')
            ax.set_xlabel("Categories")
            ax.set_ylabel("mAP", color='g')
            plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='center')

            ax2=ax.twinx()
            ax2.plot(cats, freqs, color='r')
            ax2.set_ylabel("Frequency", color='r')
            plt.title(f"{iou_type}:{map_metric}")
            
            plt.grid()
            plt.tight_layout()
    
            plt.savefig(os.path.join(dir_to_save, f"per_category_map_{iou_type}_{map_metric}.png"))
            plt.close()


def plot_off_the_shelf_metrics(off_the_shelf_metrics_file, finetuned_metrics_file):
    import csv
    with open(off_the_shelf_metrics_file) as f:
        off_the_shelf_metrics = json.load(f)
    with open(finetuned_metrics_file) as f:
        finetuned_metrics = json.load(f)
    save_dir = "comparisons"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for iou_type in off_the_shelf_metrics:
        for metric in off_the_shelf_metrics[iou_type]:
            if "large" in metric or "small" in metric or "medium" in metric:
                continue
            off_the_shelf_aps, finetuned_aps, cats = [], [], []
            f = open(os.path.join(save_dir, f"{iou_type}:{metric}.csv"), 'w')
            row = ["class", "off the shelf", "finetuned"]
            for cat in off_the_shelf_metrics[iou_type][metric]:
                if cat == "Printer": continue
                off_the_shelf_cat_ap = off_the_shelf_metrics[iou_type][metric][cat]
                finetuned_cat_ap = finetuned_metrics[iou_type][metric][cat]
                off_the_shelf_aps.append(off_the_shelf_cat_ap)
                finetuned_aps.append(finetuned_cat_ap)
                cats.append(cat)
                print(iou_type, metric, cat, off_the_shelf_cat_ap, finetuned_cat_ap)
                writer = csv.writer(f)
                row = [cat, off_the_shelf_cat_ap, finetuned_cat_ap]
                writer.writerow(row)
            

            off_the_shelf_avg = np.mean(off_the_shelf_aps)
            finetuned_avg = np.mean(finetuned_aps)
            off_the_shelf_aps.append(off_the_shelf_avg)
            finetuned_aps.append(finetuned_avg)
            row = ["average", off_the_shelf_avg, finetuned_avg]
            writer.writerow(row)
            f.close()
            print(f"Off the shelf avg: {iou_type}: {metric}", off_the_shelf_avg)
            print(f"Finetuned avg: {iou_type}: {metric}", finetuned_avg)
            # print("Relative improvement: ", (finetuned_avg - off_the_shelf_avg) / off_the_shelf_avg)
            cats.append("Average")
            index = np.arange(len(cats))
            bar_width = 0.35
            fig, ax = plt.subplots()
            summer = ax.bar(index, off_the_shelf_aps, bar_width,
                            label="Coco pretrained")

            winter = ax.bar(index + bar_width, finetuned_aps,
                            bar_width, label="Finetuned with Arena Data")

            ax.set_xlabel('Category')
            metric_name = "_".join(metric.split("_")[:-2])
            ax.set_ylabel(metric_name)
            ax.set_title(f"{iou_type}: {metric_name}")
            ax.set_xticks(index + bar_width / 2)
            ax.set_xticklabels(cats, rotation=90)
            ax.legend()
            plt.grid()
            # plt.show()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{iou_type}_{metric}.png"))
            plt.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    # Mode
    parser.add_argument("--split", dest="split", type=str,
                        default="train",
                        help="Modes: {train, validation, test}")
    parser.add_argument("--data-root", dest="data_root", type=str,
                        default=constants.DATA_ROOT,
                        help="Root to image data")
    parser.add_argument("--rg-object-list-root", dest="rg_object_list_root", type=str,
                        default=constants.RG_OBJECT_LIST_ROOT,
                        help="Root to object manifests")
    parser.add_argument("--class-to-area-thresholds-path", dest="class_to_area_thresholds_path", type=str,
                        default=constants.CLASS_TO_AREA_THRESHOLDS_PATH,
                        help="Path to precomputed class to area thresholds mapping.")
    parser.add_argument("--classes-path", dest="classes_path", type=str,
                        default=constants.CLASSES_PATH,
                        help="Path to a list of classes to include.")
    parser.add_argument("--custom-class-to-object-id-path", dest="custom_class_to_object_id_path", type=str,
                        default=constants.CUSTOM_CLASS_TO_OBJECT_ID_PATH,
                        help="Path to a customized class to object id to be used as is without any processing")
    parser.add_argument("--process-small-data-portion", dest="process_small_data_portion",
                        action='store_true',
                        help="Whether to process 50000 data points for testing the pipeline")
    
    args = parser.parse_args()
    if args.classes_path and args.custom_class_to_object_id_path:
        raise ValueError(f"Both {args.classes_path} and {args.custom_class_to_object_id_path} are provided. "
                          "These are not compatible with each other as "
                          "classes_path may filter more classes from custom_class_to_obj_ids. Make the "
                          "classes_path an empty string. Run python prepare_clean_data.py --classes-path '' ")
    data_entrypoint_save_dir = f"{args.split}_data_processed"
    args.data_entrypoint_save_dir = data_entrypoint_save_dir
    if os.path.exists(data_entrypoint_save_dir):
        raise ValueError(f"{data_entrypoint_save_dir} already exists. Please assess deletion and overwriting.")
    os.makedirs(data_entrypoint_save_dir)
    process_data(args)
