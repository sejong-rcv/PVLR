#! /usr/bin/env python
import os

if __name__ == '__main__':

    seeds = [3552]
    lrs = [0.00005]
    weight_decays = [0.001]
    batch_sizes = [10] 
    num_similars = [3] 
    similar_sizes = [2] 
    num_prob_vs = [20]
    sig_Ts = [8] 
    prefixs = [4] 
    postfixs = [4] 
    train_topks = [7] 
    alpha4s = [200] 
    alpha5s = [0.005,0.001] 
    alpha6s = [0.005,0.001] 
    alpha7s = [0.5] 
    k_easys = [5,20] 
    k_hards = [5,20] 
    Ms = [3]
    ms = [24]
    nce_Ts = [0.01] 

    # test_topks = [10] 
    # vid_threshs = [0.2] 
    # soft_nms_threshs = [0.7]
    # sigmas = [0.3] 
    # scales = [1] 
    # gamma_oics = [0.2] 

    seed = 3552
    lr = 0.00005
    weight_decay = 0.001
    batch_size = 10 
    num_similar =3 
    similar_size = 2 

    max_seqlen = 320
    optimizer = 'Adam'
    alpha1 = 0.8
    alpha2 = 0.8
    alpha3 = 1.0
    test_topk = 15
    vid_thresh = 0.2
    soft_nms_thresh = 0.7
    sigma = 0.3
    scale = 1
    gamma_oic = 0.2
    M = 3
    m = 24
    exp_id = 0
    search_id = 'metric_ablation/'

    if os.path.isdir(search_id)==False:
        os.mkdir(search_id)

    # for seed in seeds:
    #     for lr in lrs:
    #         for weight_decay in weight_decays:
    #             for batch_size in batch_sizes:
    #                 for num_similar in num_similars:
    #                     for similar_size in similar_sizes:
    
    for num_prob_v in num_prob_vs:
        for sig_T in sig_Ts:
            for prefix in prefixs:
                for postfix in postfixs:
                    for train_topk in train_topks:
                        for alpha4 in alpha4s:
                            for alpha5 in alpha5s:
                                for alpha6 in alpha6s:
                                    for alpha7 in alpha7s:
                                        for k_easy in k_easys:
                                            for k_hard in k_hards:
                                                for m in ms:
                                                    for M in Ms:
                                                        for nce_T in nce_Ts:
                                                            expname = f'seed_{seed}_lr_{lr}_weight_decay_{weight_decay}_batch_size_{batch_size}_num_similar_{num_similar}_similar_size_{similar_size}_num_prob_v_{num_prob_v}_sig_T_{sig_T}_prefix_{prefix}_postfix_{postfix}_train_topk_{train_topk}_alpha4_{alpha4}_alpha5_{alpha5}_alpha6_{alpha6}_alpha7_{alpha7}_k_easy_{k_easy}_k_hard_{k_hard}_M_{M}_m_{m}_nce_T_{nce_T}'
                                                            gpu_id = exp_id % 8
                                                            abl_id = exp_id % 16
                                                            with open (search_id + expname + '.sh', 'w') as rsh:
                                                                rsh.write(f'''\
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES={gpu_id} python main.py \\
                                --model-name {expname} \\
                                --seed {seed} \\
                                --optimizer {optimizer} \\
                                --lr {lr} \\
                                --weight_decay {weight_decay} \\
                                --max-seqlen {max_seqlen} \\
                                --batch-size {batch_size} \\
                                --num-similar {num_similar} \\
                                --similar-size {similar_size} \\
                                --num-prob-v {num_prob_v} \\
                                --sig-T {sig_T} \\
                                --prefix {prefix} \\
                                --postfix {postfix} \\
                                --train_topk {train_topk} \\
                                --alpha1 {alpha1} \\
                                --alpha2 {alpha2} \\
                                --alpha3 {alpha3} \\
                                --alpha4 {alpha4}  \\
                                --alpha5 {alpha5}  \\
                                --alpha6 {alpha6}  \\
                                --alpha7 {alpha7}  \\
                                --nce_T {nce_T} \\
                                --k_easy {k_easy}  \\
                                --k_hard {k_hard}  \\
                                --M {M}  \\
                                --m {m}  \\
                                --test_topk {test_topk} \\
                                --vid_thresh {vid_thresh} \\
                                --soft_nms_thresh {soft_nms_thresh} \\
                                --sigma {sigma} \\
                                --scale {scale} \\
                                --gamma-oic {gamma_oic} \\
                                                                ''')
                                                            with open(f'./ablation_{abl_id}.sh', 'a') as total_sh:
                                                                total_sh.write(f"bash ./experiments/{search_id}{expname}.sh\n")
                                                            exp_id += 1
