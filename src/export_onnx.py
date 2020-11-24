#coding=utf-8
# 请下载官方的代码，然后执行这个就可以生成了
import _init_paths
import numpy as np
import torch
import torch.onnx.utils as onnx
import lib.models.networks.pose_dla_dcn_to_onnx as net
from collections import OrderedDict
import cv2


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model



model = net.get_pose_net(num_layers=34, heads={'hm': 1, 'wh': 2, 'id':512, 'reg': 2})
model = load_model(model, "../models/all_dla34.pth")
model.eval()
model.cuda()
# # https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md 这里下载的
# # 如果下载不了，可以尝试我提供的连接：http://zifuture.com:1000/fs/public_models/ctdet_coco_dla_2x.pth
# checkpoint = torch.load(r"ctdet_coco_dla_2x.pth", map_location="cpu")
# checkpoint = checkpoint["state_dict"]
# change = OrderedDict()
# for key, op in checkpoint.items():
#     change[key.replace("module.", "", 1)] = op
#
# model.load_state_dict(change)
# model.eval()
# model.cuda()

input = torch.zeros((1, 3, 608, 1088)).cuda()
#
# # 有个已经导出好的模型：http://zifuture.com:1000/fs/public_models/dladcnv2.onnx
onnx.export(model, (input), "../models/all_dla34.onnx", output_names=["hm", "wh", "reg", "id", "hm_pool"], verbose=True)
#onnx.export(model, (input), "../models/all_dla34.onnx", output_names=["hm", "wh", "reg", "id"], verbose=True)
