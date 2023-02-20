import torch.nn.functional as I
import numpy as np
import cv2
import torch
from modeling.vl_model.data_generators.vl_data_generator import tokenize
import modeling.inference.constants as constants


def process_inputs_for_vl_model(image, utterance='', oracle_output=None):
    mean = torch.tensor(constants.means).reshape(3, 1, 1)
    std = torch.tensor(constants.stds).reshape(3, 1, 1)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    if not isinstance(image, torch.FloatTensor):
        image = image.float()
        image.div_(255.).sub_(mean).div_(std)
    sentences = [utterance.strip()]
    if oracle_output:
        sentences.append(oracle_output['text'].strip().lower())
    language_input = [' '.join(sentences)]
    print(sentences)
    text = tokenize(language_input, 22, True)
    image = image.cuda(non_blocking=True)
    text = text.cuda(non_blocking=True)
    return torch.unsqueeze(image, 0), text


def postprocess_mask(pred, img):
    if pred.shape[-2:] != img.shape[-2:]:
        pred = I.interpolate(
            pred,
            size=img.shape[-2:],
            mode='bicubic',
            align_corners=True).squeeze()
    mat = np.array([[1., 0., 0.],
                    [0., 1., 0.]])
    w, h = 300, 300
    pred = pred.cpu().numpy()
    pred = cv2.warpAffine(pred, mat, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderValue=0.)
    return pred
