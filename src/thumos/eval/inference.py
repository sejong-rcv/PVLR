import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from eval.classificationMAP import getClassificationMAP as cmAP
from eval.detectionMAP import getDetectionMAP as dmAP
from eval.eval_detection import ANETdetection

import os
import random
import utils.wsad_utils as utils
import numpy as np
import pandas as pd
from collections import defaultdict
import scipy.io as sio
import multiprocessing as mp
from tabulate import tabulate

import wsad_dataset
import options
import model
from model.prompt import text_prompt
from model import wstal
import proposal_methods as PM

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


torch.set_default_tensor_type('torch.cuda.FloatTensor')

def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

@torch.no_grad()
def test(itr, dataset, args, model, device):
   model.eval()
   done = False
   X_gmm = []
   y_gmm = []
   instance_logits_stack = []
   labels_stack = []
   proposals = []
   results = defaultdict(dict)
   while not done:
      features, clip_feature, temp_anno, labels, vn, done = dataset.load_data(is_training=False) 
      if features.shape[0] != clip_feature.shape[0]:
         clip_feature = clip_feature[:features.shape[0]]
      seq_len = [features.shape[0]]
      if seq_len == 0:
         continue
      features = torch.from_numpy(features).float().to(device).unsqueeze(0)
      clip_feat = torch.from_numpy(clip_feature).float().to(device).unsqueeze(0)

      with torch.no_grad():
         outputs = model(Variable(features), clip_feat, split='test', itr=itr, opt=args)
         element_logits = outputs['cas']
         proposals.append(getattr(PM, args.proposal_method)(args, vn, outputs, labels)) 
         logits=element_logits.squeeze(0)

      tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
      instance_logits_stack.append(tmp)
      labels_stack.append(labels)

   instance_logits_stack = np.array(instance_logits_stack)
   labels_stack = np.array(labels_stack)
   proposals = pd.concat(proposals).reset_index(drop=True)

   #CVPR2020
   if 'Thumos14' in args.dataset_name:
      iou = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
      dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args)
      dmap_detect.prediction = proposals
      dmap,dap = dmap_detect.evaluate()
   else:
      iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95]

      dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args,subset='val')
      dmap_detect.prediction = proposals
      dmap,dap = dmap_detect.evaluate()

   if args.dataset_name == 'Thumos14':
      test_set = sio.loadmat('test_set_meta.mat')['test_videos'][0]
      for i in range(np.shape(labels_stack)[0]):
         if test_set[i]['background_video'] == 'YES':
               labels_stack[i,:] = np.zeros_like(labels_stack[i,:])

   cmap = cmAP(instance_logits_stack, labels_stack)
   print('Classification map %f' %cmap)
   return iou,dmap,dap

if __name__ == '__main__':
    classes = {
            'BaseballPitch': 'baseball pitch',
            'BasketballDunk': 'basketball dunk',
            'Billiards': 'billiards',
            'CleanAndJerk': 'clean and jerk',
            'CliffDiving': 'cliff diving',
            'CricketBowling': 'cricket bowling',
            'CricketShot': 'cricket shot',
            'Diving': 'diving',
            'FrisbeeCatch': 'frisbee catch',
            'GolfSwing': 'golf swing',
            'HammerThrow': 'hammer throw',
            'HighJump': 'high jump',
            'JavelinThrow': 'javelin throw',
            'LongJump': 'long jump',
            'PoleVault': 'pole vault',
            'Shotput': 'shot put',
            'SoccerPenalty': 'soccer penalty',
            'TennisSwing': 'tennis swing',
            'ThrowDiscus': 'throw discus',
            'VolleyballSpiking': 'volleyball spiking'
            }
    inp_actionlist = list(classes.values())

    args = options.parser.parse_args()
    device = torch.device("cuda")

    seed=args.seed
    setup_seed(seed)

    if os.path.isdir(os.path.join("output","log_inference")) == False:
       os.mkdir(os.path.join("output","log_inference"))
    
    if os.path.isdir(os.path.join("output","log_inference",args.model_name)) == False:
       os.mkdir(os.path.join("output","log_inference",args.model_name))
    result_file = open(os.path.join("output","log_inference",args.model_name,'Performance.txt'),'w')

    dataset = getattr(wsad_dataset, args.dataset)(args)
    actionlist, actiondict, actiontoken = text_prompt(dataset=args.dataset_name, clipbackbone=args.backbone, device=device)
    PVLR = wstal.PVLR(actiondict=actiondict, actiontoken=actiontoken, inp_actionlist=inp_actionlist, opt=args).to(device)

    assert (args.pretrained_ckpt is not None)
    PVLR.load_state_dict(torch.load(args.pretrained_ckpt))
    print(f"CKPT loaded: '{args.pretrained_ckpt}'")
    
    iou, dmap, dap = test(0, dataset, args, PVLR, device)

    columns = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.3:0.7', '0.1:0.7'] 
    dmap = (dmap*100)[:7]
    
    performance = [dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6], np.mean(dmap[2:7]), np.mean(dmap[:7])]

    table = [performance]
    print(tabulate(table, headers=columns, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'))
    print(args.model_name + "\n", file=result_file, flush=True)
    print(tabulate(table, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'), file=result_file, flush=True)

