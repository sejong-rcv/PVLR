import sys
sys.path.append('./')

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from eval.classificationMAP import getClassificationMAP as cmAP
from eval.detectionMAP import getDetectionMAP as dmAP
from eval.eval_detection import ANETdetection

import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from tabulate import tabulate

import wsad_dataset
import options
import model
from model.prompt import text_prompt
from model import wstal
import proposal_methods as PM


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
   instance_logits_stack = []
   element_logits_stack = []
   labels_stack = []
   proposals = []
   results = defaultdict(dict)

   while not done:
      features, clip_feature, labels, vn, done = dataset.load_data(is_training=False)
      if features.shape[0] != clip_feature.shape[0]:
         clip_feature = clip_feature[:features.shape[0]]

      seq_len = [features.shape[0]]
      if seq_len == 0:
         continue
      features = torch.from_numpy(features).float().to(device).unsqueeze(0)
      clip_feat = torch.from_numpy(clip_feature).float().to(device).unsqueeze(0)

      with torch.no_grad():
         outputs = model(Variable(features), clip_feat, is_training=False, seq_len=seq_len)
         element_logits = outputs['cas']
         results[vn] = {'cas':outputs['cas'], 'attn':outputs['attn']}

         proposals.append(getattr(PM, args.proposal_method)(vn, outputs, args))
         logits=element_logits.squeeze(0)
      tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()

      instance_logits_stack.append(tmp)
      labels_stack.append(labels)

   instance_logits_stack = np.array(instance_logits_stack)
   labels_stack = np.array(labels_stack)
   proposals = pd.concat(proposals).reset_index(drop=True)

   iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
   dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='val')
   dmap_detect.prediction = proposals
   dmap = dmap_detect.evaluate()
    
   cmap = cmAP(instance_logits_stack, labels_stack)
   print('Classification map %f' %cmap)
    
   return iou, dmap

def classes(cls):
    return cls.lower()

if __name__ == '__main__':
   meta = np.load("../../data/annet/ActivityNet1.3/ActivityNet1.3-Annotations/classlist.npy", 'r')
   inp_actionlist = [classes(act) for act in meta]
   
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
   PVLR = wstal.PVLR(dataset.feature_size, dataset.num_class, actiondict=actiondict, actiontoken=actiontoken, inp_actionlist=inp_actionlist, opt=args).to(device)

   assert (args.pretrained_ckpt is not None)
   PVLR.load_state_dict(torch.load(args.pretrained_ckpt))
   print(f"CKPT loaded: '{args.pretrained_ckpt}'")
   
   iou, dmap = test(0, dataset, args, PVLR, device)

   columns = ['0.5', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '0.5:0.95']
   dmap = (dmap*100)
   
   performance = [dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6], dmap[7], dmap[8], dmap[9], np.mean(dmap)]
   table = [performance]
   print(tabulate(table, headers=columns, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'))
   print(tabulate(table, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'), file=result_file, flush=True)