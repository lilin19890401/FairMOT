# coding=utf-8
from __future__ import division
from pprint import pprint
import logging
import numpy as np
import onnx
import onnxruntime
import torch
import torch.onnx
import cv2
from utils.image import get_affine_transform
from utils.post_process import ctdet_post_process
from models.decode import mot_decode


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

mean = [0.408, 0.447, 0.470]  # coco and kitti not same
std = [0.289, 0.274, 0.278]
down_ratio = 4
test_scales = [1]


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad \
        else tensor.cpu().numpy()

def pre_process(image, scale, meta=None):
    height, width = image.shape[0:2]
    new_height = int(height * scale)
    new_width = int(width * scale)

    inp_height, inp_width = (512, 512)
    c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
    s = max(height, width) * 1.0

    trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
    resized_image = cv2.resize(image, (new_width, new_height))
    inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
    inp_image = ((inp_image / 255. - mean) /
                 std).astype(np.float32)

    images = inp_image.transpose(2, 0, 1).reshape(
        1, 3, inp_height, inp_width)
    ##images = torch.from_numpy(images)
    meta = {'c': c, 's': s,
            'out_height': inp_height // down_ratio,
            'out_width': inp_width // down_ratio}
    return images, meta


if __name__ == "__main__":
    onnx_path = r'D:\DeepLearning\ObjectTrackingMethod\FairMOT\models\all_dla34.onnx'

    # load onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    img = cv2.imread(r'F:\图片\me.PNG')
    images, meta = pre_process(img, 1, None)
    #images = images.to(device)

    # forward onnx model
    ort_session = onnxruntime.InferenceSession(onnx_path)

    for ii in ort_session.get_inputs():
        print('onnx input: {}'.format(ii.name))
    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(image).astype(np.float32)}
    ort_inputs = {ort_session.get_inputs()[0].name: images}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])
    # logger.info("ONNX model forwarded successfully")

    # 后处理
    # heads = {'hm': 80, 'reg': 2, 'wh': 2}
    hm = torch.from_numpy(ort_outs[0]).sigmoid_()
    wh = torch.from_numpy(ort_outs[2])
    reg = torch.from_numpy(ort_outs[1])
    dets = mot_decode(hm, wh, reg=reg, K=100)
    print(dets)