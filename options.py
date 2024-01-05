import argparse

parser = argparse.ArgumentParser(description='TSM-NET')

# Default Setting
parser.add_argument('--path-dataset', type=str, default='path/to/Thumos14', help='the path of data feature')
parser.add_argument('--path-clip-dataset', type=str, default='../clip_feature/clip_feature_RN50/mid', help='the path of data feature')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--dataset-name', default='Thumos14reduced', help='dataset to train on (default: )')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use-model',type=str, default='CO2', help='model used to train the network')
parser.add_argument('--backbone', type=str, default='RN50', choices=['ViT-B/16','RN50'])
parser.add_argument('--dataset',type=str,default='SampleDataset')
parser.add_argument('--proposal_method',type=str,default='multiple_threshold_hamnet')

parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int,default=20, help='number of classes (default: )')
parser.add_argument('--max-iter', type=int, default=9001, help='maximum iteration to train (default: 50000)')
parser.add_argument('--interval', type=int, default=50,help='time interval of performing the test')
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--warmup_iter', type=int, default=3000, help='time interval of performing the test')

# Argument(Grid-Search)
# main
parser.add_argument('--seed', type=int, default=3552)
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD'])
parser.add_argument('--lr', type=float, default=5e-05)
parser.add_argument('--weight_decay', type=float, default=0.001)

# dataset
parser.add_argument('--max-seqlen', type=int, default=320)
parser.add_argument('--batch-size', type=int, default=10)
parser.add_argument('--num-similar', type=int, default=3)
parser.add_argument('--similar-size', type=int, default=2)

# model
parser.add_argument('--num-prob-t', type=int, default=1)
parser.add_argument('--num-prob-v', type=int, default=20)
parser.add_argument('--sig-T_train', type=float, default=8)
parser.add_argument('--sig-T_infer', type=float, default=0.03)
parser.add_argument('--prefix', type=int, default=4)
parser.add_argument('--postfix', type=int, default=4)

# loss
parser.add_argument('--train_topk',type=int,default=7)
parser.add_argument('--k_easy',type=int,default=20)
parser.add_argument('--k_hard',type=int,default=5)

parser.add_argument('--M',type=int,default=3)
parser.add_argument('--m',type=int,default=24)
parser.add_argument('--nce_T',type=float,default=0.01)
parser.add_argument('--init_shift',type=float,default=0.01)
parser.add_argument('--init_negative_scale',type=float,default=15)
parser.add_argument('--metric',type=str, default='KL_div', choices=['KL_div','Bhatta','Mahala', 'Euclidean'])

parser.add_argument("--alpha0", type=float, default=0.8) 
parser.add_argument("--alpha1", type=float, default=0.8) 
parser.add_argument("--alpha2", type=float, default=0.8) 
parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4',type=float,default=200)
parser.add_argument('--alpha5',type=float,default=0.005)
parser.add_argument('--alpha6',type=float,default=0.001)
parser.add_argument('--alpha7',type=float,default=0.5)

# test
parser.add_argument('--test_topk',type=int,default=20)
parser.add_argument('--vid_thresh',type=float,default=0.2)
parser.add_argument('--soft_nms_thresh',type=float,default=0.7)
parser.add_argument('--sigma',type=float,default=0.3)
parser.add_argument('--scale',type=float,default=1)
parser.add_argument('--gamma-oic', type=float, default=0.1)

parser.add_argument('--act-s', type=float, default=0.1)
parser.add_argument('--act-e', type=float, default=0.95)
parser.add_argument('--cas-s', type=float, default=0.0)
parser.add_argument('--cas-e', type=float, default=0.95)
parser.add_argument('--act-num',type=int,default=20)
parser.add_argument('--cas-num',type=int,default=20)




