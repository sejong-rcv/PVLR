#! /usr/bin/env python

a4s = [10, 5, 2, 1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001]
# total_num_exps = len(a1s) * len(a2s) * len(a3s) * len(a4s)
exp_id = 0

# for a1 in a1s:
#         for a2 in a2s:
#             for a3 in a3s:
#                 for idx, a4 in enumerate(a4s):
#                     expname = f'a1_{a1}_a2_{a2}_a3_{a3}_a4_{a4}'
#                     gpu_id = exp_id % 4
#                     with open ('/home/hwkim/workspace/access_2024/video_prob_clip/experiments/'+expname+'.sh', 'w') as rsh:
#                         rsh.write(f'''\
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gpu_id} python main.py \\
#                                                 --max-seqlen 320 \\
#                                                 --lr 0.00005 \\
#                                                 --k 7 \\
#                                                 --dataset-name Thumos14reduced \\
#                                                 --num-class 20 \\
#                                                 --use-model CO2 \\
#                                                 --max-iter 6001 \\
#                                                 --dataset SampleDataset \\
#                                                 --weight_decay 0.001 \\
#                                                 --model-name {expname} \\
#                                                 --alpha4 {a4} \\
#                                                 --num-prob-v 20 \\
#                                                 --sig-T 1 \\
#                                                 --seed 1''')
#                     with open(f'/home/hwkim/workspace/access_2024/video_prob_clip/experiments/ablation_{gpu_id}.sh', 'a') as total_sh:
#                         total_sh.write(f"bash ./experiments/{expname}.sh\n")
#                     exp_id += 1


for idx, a4 in enumerate(a4s):
    expname = f'a4_{a4}'
    gpu_id = exp_id % 4
    with open ('/home/hwkim/workspace/access_2024/video_prob_clip_loss_ablation/experiments/'+expname+'.sh', 'w') as rsh:
        rsh.write(f'''\
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gpu_id} python main.py \\
                                --max-seqlen 320 \\
                                --lr 0.00005 \\
                                --k 7 \\
                                --dataset-name Thumos14reduced \\
                                --num-class 20 \\
                                --use-model CO2 \\
                                --max-iter 6001 \\
                                --dataset SampleDataset \\
                                --weight_decay 0.001 \\
                                --model-name {expname} \\
                                --alpha4 {a4} \\
                                --num-prob-v 20 \\
                                --sig-T 1 \\
                                --seed 1''')
    with open(f'/home/hwkim/workspace/access_2024/video_prob_clip_loss_ablation/experiments/ablation_{gpu_id}.sh', 'a') as total_sh:
        total_sh.write(f"bash ./experiments/{expname}.sh\n")
    exp_id += 1