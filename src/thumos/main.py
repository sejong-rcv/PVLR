from __future__ import print_function
import os
import random
import json
import pickle
import tarfile

import numpy as np
import pandas as pd
import scipy.io as sio
import multiprocessing as mp

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import options
import dataset

from model.prompt import text_prompt
from model.loss import TotalLoss
from model import wstal

from eval import proposal_methods as PM
from eval.eval_detection import ANETdetection
from eval.classificationMAP import getClassificationMAP as cmAP
from eval.detectionMAP import getDetectionMAP as dmAP

from tqdm import tqdm
from tabulate import tabulate
from collections import defaultdict

torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
   features, clip_feature, temp_anno, labels, pairs_id = dataset.load_data(n_similar=args.num_similar)
   seq_len = np.sum(np.max(np.abs(features), axis=2) > 0, axis=1)
   features = features[:,:np.max(seq_len),:]
   features = torch.from_numpy(features).float().to(device)
   temp_anno = torch.from_numpy(temp_anno).float().to(device)
   labels = torch.from_numpy(labels).float().to(device)
   clip_feature = torch.from_numpy(clip_feature).float().to(device)
   
   outputs = model(features,clip_feature,itr=itr,split='train',device=device,opt=args)

   loss, loss_dict = criterion(itr, outputs, clip_feature, labels)

   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   return loss.data.cpu().numpy(), loss_dict

@torch.no_grad()
def test(itr, dataset, args, model, device):

   if os.path.isdir(os.path.join("output","results",args.model_name))==False:
      os.mkdir(os.path.join("output","results",args.model_name))

   model.eval()
   done = False
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

         results[vn.decode('utf-8')] = {'cas':outputs['cas'].detach().cpu().numpy(), 
                                       'attn':outputs['attn'].detach().cpu().numpy(),
                                       'reparam_sim':outputs['reparm_sim'].detach().cpu().numpy(),
                                       'distill_sim':outputs['distill_sim'].detach().cpu().numpy() 
                                       }

         proposals.append(getattr(PM, args.proposal_method)(args, vn, outputs, labels))
         logits=element_logits.squeeze(0)

      tmp = F.softmax(torch.mean(torch.topk(logits, k=int(np.ceil(len(features)/8)), dim=0)[0], dim=0), dim=0).cpu().data.numpy()
      instance_logits_stack.append(tmp)
      labels_stack.append(labels)


   instance_logits_stack = np.array(instance_logits_stack)
   labels_stack = np.array(labels_stack)
   proposals = pd.concat(proposals).reset_index(drop=True)
   
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

   results['cmap'] = cmap
   results['dmap'] = dmap
   results['dap'] = dap

   with open(os.path.join("output","results",args.model_name,f"iter_{itr}_results.pkl"), 'wb') as f:
      pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

   return iou,dmap,dap

if __name__ == '__main__':
   pool = mp.Pool(20)
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
   dataset = getattr(dataset, args.dataset)(args)

   if 'Thumos' in args.dataset_name:
      max_map=[0]*9
   else:
      max_map=[0]*10

   if not os.path.exists('output/ckpt/'):
      os.makedirs('output/ckpt/')

   if not os.path.exists(f'output/ckpt/{args.model_name}'):
      os.makedirs(f'output/ckpt/{args.model_name}')

   save_dir = f'output/ckpt/{args.model_name}'
   actionlist, actiondict, actiontoken = text_prompt(dataset=args.dataset_name, clipbackbone=args.backbone, device=device)
   TSM = wstal.TSM(actiondict=actiondict, actiontoken=actiontoken, inp_actionlist=inp_actionlist, opt=args).to(device)

   if args.pretrained_ckpt is not None:
      previous_model = torch.load(args.pretrained_ckpt)
      wts = ["fc_clip.weight", "fc_clip.bias", "text_prob_encoder.fc_mean.weight", "text_prob_encoder.fc_mean.bias", \
             "text_prob_encoder.fc_var.weight", "text_prob_encoder.fc_var.bias", "text_prob_encoder.layer_norm.weight", \
             "text_prob_encoder.layer_norm.bias", "snippet_prob_encoder.layer_norm.weight", "snippet_prob_encoder.layer_norm.bias"]
      for wt in wts:
         del previous_model[wt]
      TSM.load_state_dict(previous_model)
      print("Original ckpt loaded !!!")

   optimizer = optim.Adam([{"params": TSM.parameters()}], lr=args.lr, weight_decay=args.weight_decay)
   criterion = TotalLoss(args)

   total_loss = 0
   best_mean = 0
   best_performance = []

   for itr in tqdm(range(args.max_iter)):
      loss, loss_dict = train(itr, dataset, args, TSM ,optimizer, criterion, device)
      total_loss+=loss

      if itr > args.warmup_iter and itr % args.interval == 0 and not itr == 0:  
         print('Iteration: %d, Loss: %.5f ' %(itr, total_loss/args.interval))
         total_loss = 0

         iou, dmap, dap = test(itr, dataset, args, TSM, device)
         if 'Thumos' in args.dataset_name:
            cond=sum(dmap[0:7])>sum(max_map[0:7])
         else:
            cond=np.mean(dmap)>np.mean(max_map)
            
         if cond:
            torch.save(TSM.state_dict(), f'output/ckpt/{args.model_name}/Best_model.pkl')
            max_map = dmap

         columns = [' ', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.3:0.7', '0.1:0.7'] 
         dmap = (dmap*100)[:7]
         
         performance = [itr, dmap[0], dmap[1], dmap[2], dmap[3], dmap[4], dmap[5], dmap[6], np.mean(dmap[2:7]), np.mean(dmap[:7])]
         if np.mean(dmap[:7]) > best_mean:
            best_mean = np.mean(dmap[:7])
            best_performance = performance
         table = [best_performance, performance]
         print(tabulate(table, headers=columns, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'))
         print(tabulate(table, numalign="center", stralign="center", tablefmt="simple", floatfmt='.2f'), file=result_file, flush=True)