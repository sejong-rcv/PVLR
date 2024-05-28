import argparse

parser = argparse.ArgumentParser(description='CO2-NET')

# Default Setting
parser.add_argument('--path-dataset', type=str, default='features/ActivityNet1.3', help='the path of data feature')
parser.add_argument('--path-clip-dataset', type=str, default='features/Anet_CLIP', help='the path of data feature')
parser.add_argument('--model-name', default='weakloc', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--dataset-name', default='ActivityNet1.3', help='dataset to train on (default: )')
parser.add_argument('--feature-type', type=str, default='I3D', help='type of feature to be used I3D or UNT (default: I3D)')
parser.add_argument('--use-model',type=str, default='CO2', help='model used to train the network')

parser.add_argument('--backbone', type=str, default='RN50', choices=['ViT-B/16','RN50'])
parser.add_argument('--dataset',type=str,default='AntSampleDataset')
parser.add_argument('--proposal_method',type=str,default='multiple_threshold_hamnet')
parser.add_argument('--feature-size', default=2048, help='size of feature (default: 2048)')
parser.add_argument('--num-class', type=int, default=200, help='number of classes (default: 20)')
parser.add_argument('--max-iter', type=int, default=20001, help='maximum iteration to train (default: 50000)')
parser.add_argument('--interval', type=int, default=1000,help='time interval of performing the test')
parser.add_argument("--feature_fps", type=int, default=25)
parser.add_argument('--warmup_iter', type=int, default=10000, help='time interval of performing the test')
parser.add_argument('--seed', type=int, default=777, help='random seed (default: 1)')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam','SGD'])
parser.add_argument('--lr', type=float, default=0.00003,help='learning rate (default: 0.0001)')
parser.add_argument('--weight_decay', type=float, default=5e-5)
parser.add_argument('--max-seqlen', type=int, default=60, help='maximum sequence length during training (default: 750)')
parser.add_argument('--batch-size', type=int, default=10, help='number of instances in a batch of data (default: 10)')
parser.add_argument('--num-similar', default=3, type=int,help='number of similar pairs in a batch of data  (default: 3)')
parser.add_argument('--similar-size', type=int, default=2)

# model
parser.add_argument('--num-prob-v', type=int, default=20)
parser.add_argument('--train-sig-T', type=float, default=4)
parser.add_argument('--test-sig-T', type=float, default=0.003)
# parser.add_argument('--sig-T-attn', type=float, default=1.0)
parser.add_argument('--prefix', type=int, default=4)
parser.add_argument('--postfix', type=int, default=4)
parser.add_argument('--convex_alpha', type=float, default=0.6)
parser.add_argument('--eps-std', type=float, default=0.005)
parser.add_argument('--train_topk',type=float,default=5)
parser.add_argument('--k_easy',type=int,default=3)
parser.add_argument('--k_hard',type=int,default=3)
parser.add_argument('--M',type=int,default=3)
parser.add_argument('--m',type=int,default=12)
parser.add_argument('--metric',type=str, default='KL_div', choices=['KL_div','Bhatta','Mahala'])
# parser.add_argument('--metric_vid',type=str, default='KL_div', choices=['cos', 'KL_div'])
# parser.add_argument('--loss_type',type=str, default='neg_log', choices=['frobenius', 'neg_log'])
# parser.add_argument('--var_type',type=str, default='definition', choices=['naive_weighted', 'definition'])

# loss
parser.add_argument("--alpha0", type=float, default=0.8)
parser.add_argument("--alpha1", type=float, default=0.8)
parser.add_argument("--alpha2", type=float, default=0.8)
parser.add_argument("--alpha3", type=float, default=1)
parser.add_argument('--alpha4',type=float,default=1)
parser.add_argument('--alpha5',type=float,default=0.01)
parser.add_argument('--alpha6',type=float,default=0.01)
parser.add_argument('--alpha7',type=float,default=4)
parser.add_argument('--alpha8',type=float,default=0.01)

# test
# parser.add_argument('--test-topk',type=int,default=20)
parser.add_argument('--vid_threshold', type=float, default=0.2)
parser.add_argument("--soft_nms_thresh", type=float, default=0.7)
# parser.add_argument('--gamma-oic', type=float, default=0.1)
parser.add_argument("--sigma", type=float, default=0.5)
parser.add_argument('--cas_scale',type=float,default=1)
parser.add_argument('--cas_gamma-oic', type=float, default=0.01)
parser.add_argument('--cas_lambda_',type=float,default=0.2)
parser.add_argument('--act_scale',type=float,default=1)
parser.add_argument('--act_gamma-oic', type=float, default=0.5)
parser.add_argument('--act_lambda_',type=float,default=0.4)
parser.add_argument('--act-s', type=float, default=0.2)
parser.add_argument('--act-e', type=float, default=0.95)
parser.add_argument('--cas-s', type=float, default=0.1)
parser.add_argument('--cas-e', type=float, default=0.95)
parser.add_argument('--act-num',type=int,default=10)
parser.add_argument('--cas-num',type=int,default=10)

# etc..? need to check
parser.add_argument("--topk2", type=float, default=10)
parser.add_argument("--topk", type=float, default=60)
parser.add_argument("--ckpt_iter", type=int, default=18000)
parser.add_argument('--dropout_ratio',type=float,default=0.7)
parser.add_argument('--reduce_ratio',type=int,default=16)
parser.add_argument('--t',type=int,default=5)
parser.add_argument("--AWM", type=str, default='BWA_fusion_dropout_feat_v2')