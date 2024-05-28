from __future__ import print_function
import argparse
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import json
from tabulate import tabulate
from tqdm import tqdm
from collections import defaultdict
import multiprocessing as mp

from model import wstal
from model.prompt import text_prompt
from model.loss import TotalLoss
import wsad_dataset
import options

from eval import proposal_methods as PM
from eval.eval_detection import ANETdetection
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.detectionMAP import getDetectionMAP as dmAP

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def classes(cls):
   return cls.lower()
meta = np.load("features/ActivityNet1.3/ActivityNet1.3-Annotations/classlist.npy", 'r')
inp_actionlist = [classes(act) for act in meta]

def setup_seed(seed):
   random.seed(seed)
   os.environ['PYTHONHASHSEED'] = str(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.deterministic = True

def train(itr, dataset, args, model, optimizer, criterion, device):
    model.train()
    features, clip_feature, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
    seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
    features = features[:,:np.max(seq_len),:]
    
    features = torch.from_numpy(features).float().to(device)
    clip_feature = torch.from_numpy(clip_feature).float().to(device)
    labels = torch.from_numpy(labels).float().to(device)
    
    pseudo_label = None
    outputs = model(features, clip_feature, seq_len=seq_len, is_training=True, itr=itr, opt=args)

    total_loss, loss_dict = criterion(itr, outputs, clip_feature, labels)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return total_loss.data.cpu().numpy(), loss_dict

@torch.no_grad()
def test(itr, dataset, args, model, device, pool):
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

   iou = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,0.95]
   dmap_detect = ANETdetection(dataset.path_to_annotations, iou, args=args, subset='val')
   dmap_detect.prediction = proposals
   dmap = dmap_detect.evaluate()
    
   cmap = cmAP(instance_logits_stack, labels_stack)
   print('Classification map %f' %cmap)
    
   return iou, dmap

if __name__ == '__main__':
   pool = mp.Pool(5)

   args = options.parser.parse_args()
   args_dict = vars(args)

   seed=args.seed
   setup_seed(seed)
   print('=============seed: {}, pid: {}============='.format(seed,os.getpid()))

   result_path = f'output/log/{args.model_name}/'
   os.makedirs(result_path, exist_ok=True)
   result_file = open(os.path.join(result_path, 'Performance.txt'),'w')
   with open(os.path.join(result_path, 'opts.json'), 'w') as j:
      json.dump(args_dict, j, indent=2)

   device = torch.device("cuda")
   dataset = getattr(wsad_dataset,args.dataset)(args)

   max_map=[0]*10

   if not os.path.exists('output/ckpt/'):
      os.makedirs('output/ckpt/')
   if not os.path.exists(f'output/ckpt/{args.model_name}'):
      os.makedirs(f'output/ckpt/{args.model_name}')
   if not os.path.exists('output/log/'):
      os.makedirs('output/log/')
   if not os.path.exists('./output/log/' + args.model_name):
      os.makedirs('./output/log/' + args.model_name)

   actionlist, actiondict, actiontoken = text_prompt(dataset=args.dataset_name, clipbackbone=args.backbone, device=device)
   model = wstal.PECR(dataset.feature_size, dataset.num_class, actiondict=actiondict, actiontoken=actiontoken, inp_actionlist=inp_actionlist, opt=args).to(device)

   if args.pretrained_ckpt is not None:
      model.load_state_dict(torch.load(args.pretrained_ckpt))

   optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   criterion = TotalLoss(args)

   total_loss = 0
   best_mean = 0

   pbar = tqdm(range(args.max_iter))
   for itr in pbar:
      loss, loss_dict = train(itr, dataset, args, model, optimizer, criterion, device)
      total_loss+=loss

      if itr >= args.warmup_iter and itr % args.interval == 0 and not itr == 0:
         print('Iteration: %d, Loss: %.5f' %(itr, total_loss/args.interval))
         total_loss = 0

         iou,dmap = test(itr, dataset, args, model, device,pool)
         
         cond=np.mean(dmap)>np.mean(max_map)
         if cond:
            torch.save(model.state_dict(), f'output/ckpt/{args.model_name}/Best_model.pkl')
            max_map = dmap

         columns = [' ', '0.5', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95', '0.5:0.95']
         dmap = (dmap*100)
         
         performance = [itr, dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6], dmap[7], dmap[8], dmap[9], np.mean(dmap)]
         if np.mean(dmap) > best_mean:
            best_mean = np.mean(dmap)
            best_performance = performance
         table = [best_performance, performance]
         print(tabulate(table, headers=columns, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'))
         print(tabulate(table, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'), file=result_file, flush=True)