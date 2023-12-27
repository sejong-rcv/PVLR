import numpy as np
import pandas as pd
import json
import os
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_unique_labels_in_video(all_gt):
    total_list = []
    for anno in all_gt:
        cls = anno['label']
        total_list.append(classmap[cls])
    return set(total_list)

def get_gt_of_cur_cls(cls, all_gt):
    total_gt = []
    for anno in all_gt:
        if classmap[anno['label']] == cls:
            total_gt.append(anno['segment'])
    return total_gt

def iou(p_s, p_e, g_s, g_e):
    # import pdb;pdb.set_trace()
    p_s*=100
    p_e*=100
    g_s*=100
    g_e*=100
    gt = range(int(g_s), int(g_e))
    pred = range(int(p_s), int(p_e))
    iou = float(len(set(gt).intersection(set(pred)))) / float(len(set(gt).union(set(pred))))
    return iou

def get_iou_matrix(pred_of_cur_cls, gt_of_cur_cls):
    iou_matrix = np.zeros((len(pred_of_cur_cls), len(gt_of_cur_cls)))
    for i, pred in enumerate(pred_of_cur_cls.iterrows()):
        p_s = pred[1]['t-start']
        p_e = pred[1]['t-end']
        for j, gg in enumerate(gt_of_cur_cls):
            g_s = float(gg[0])
            g_e = float(gg[1])
            iou_matrix[i][j] = iou(p_s, p_e, g_s, g_e)
    return iou_matrix

def match_pred_with_gt(iou_matrix):
    # IoU argmax idx의 gt와 매칭 idx 넣어줄 vector
    pred_gt_match = np.zeros(iou_matrix.shape[0])
    pred_gt_match_iou = np.zeros(iou_matrix.shape[0]) ###
    # 아무 gt와도 매칭되지 않은 pred에는 -1을 넣어줌
    unmatched_idx = np.where(np.sum(iou_matrix, axis=1) == 0)[0]
    # 어떠한 gt와 매칭된 pred, argmax idx 넣어줌
    matched_idx = np.where(np.sum(iou_matrix, axis=1) != 0)[0]
    
    for match in matched_idx:
        matched_gt = iou_matrix[match].argmax()
        matched_gt_iou = iou_matrix[match].max()
        pred_gt_match[match] = matched_gt
        pred_gt_match_iou[match] = matched_gt_iou
    pred_gt_match[unmatched_idx] = -1
    
    return pred_gt_match, pred_gt_match_iou

def check_completeness(pred_of_cur_cls, gt_of_cur_cls, matched_result, matched_iou):
    cur_tp = 0
    cur_overcom = 0
    cur_incom = 0
    cur_shift = 0
    cur_fp = 0
    
    tp_iou_thr = 0.5
    cur_tp_iou = 0
    cur_overcom_iou = 0
    cur_incom_iou = 0
    cur_shift_iou = 0
    
    cur_fp += len(np.where(matched_result == -1)[0])
    matched_idx = np.where(matched_result != -1)[0]
    pred_list = pred_of_cur_cls.reset_index()
    
    for match in matched_idx:
        cur_iou = matched_iou[match]
        matched_gt_idx = int(matched_result[match])
        g_s = float(gt_of_cur_cls[matched_gt_idx][0])
        g_e = float(gt_of_cur_cls[matched_gt_idx][1]) 
        p_s = pred_list.iloc[match]['t-start']
        p_e = pred_list.iloc[match]['t-end']
        
        # proposal type과 해당 type에서의 iou
        if cur_iou >= tp_iou_thr:
            cur_tp += 1
            cur_tp_iou += cur_iou
        elif p_s<g_s and g_e<p_e:
            cur_overcom += 1
            cur_overcom_iou += cur_iou
        elif g_s<p_s and p_e<g_e:
            cur_incom += 1
            cur_incom_iou += cur_iou
        else:
            cur_shift += 1
            cur_shift_iou += cur_iou
    
    tp_avg = cur_tp_iou / cur_tp if cur_tp != 0 else 0
    overcom_avg = cur_overcom_iou / cur_overcom if cur_overcom != 0 else 0
    incom_avg = cur_incom_iou / cur_incom if cur_incom != 0 else 0
    shift_avg = cur_shift_iou / cur_shift if cur_shift != 0 else 0
    
    return cur_tp, cur_overcom, cur_incom, cur_shift, cur_fp, \
           tp_avg, overcom_avg, incom_avg, shift_avg


if __name__ == '__main__':
    # 모델의 예측값
    boosting_path = './text_embedding'
    train_pred = pd.read_csv(os.path.join(boosting_path, 'tsm_only_clip.csv'))
    cal_pred = pd.read_csv(os.path.join(boosting_path, 'tsm_only_clip.csv'))

    cal_pred['t-start'] = train_pred['t-start'] * 16 / 25
    cal_pred['t-end'] = train_pred['t-end'] * 16 / 25
    cal_pred['score'] = sigmoid(train_pred['score'])
    cal_pred.loc[cal_pred['video-id']=='video_validation_0000051']

    # GT label
    with open('/data/gtlim/workspace/[2024][Paper]WSTAL/Thumos14reduced/gt.json') as j:
        gt = json.load(j)
    gt = gt['database']
    gt['video_validation_0000051']

    # Classlist
    classlist = np.load("./Thumos14reduced-Annotations/classlist.npy")
    classlist = classlist.astype(str)
    classmap = {k : v for v, k in enumerate(classlist)}
    classmap

    # Train video list
    train_list = []
    with open('./split_train.txt') as f:
        tr = f.readlines()
    for vid in tr:
        train_list.append(vid.strip())
    train_list

    # start
    tp = 0
    overcom = 0
    incom = 0
    shift = 0
    fp = 0
    total = 0
    true_vid_total_pred = 0
    tp_iou = []
    overcom_iou = []
    incom_iou = []
    shift_iou = []

    for vid in tqdm(train_list):
        all_gt = gt[vid]['annotations']
        all_pred = cal_pred.loc[cal_pred['video-id'] == vid]
        true_vid_total_pred += all_pred.shape[0]
        labels_in_video = get_unique_labels_in_video(all_gt)
        
        for cls in labels_in_video:
            # 현재 보고 있는 클래스의 예측값과 gt값 가져옴
            pred_of_cur_cls = all_pred.loc[all_pred['label'] == cls]
            gt_of_cur_cls = get_gt_of_cur_cls(cls, all_gt)
            iou_matrix = get_iou_matrix(pred_of_cur_cls, gt_of_cur_cls)
            
            # 각 proposal이 매칭된 gt index
            # 매칭되었으면 gt index, 안되었으면 -1 값 가짐
            matched_result, matched_iou = match_pred_with_gt(iou_matrix)
            cur_tp, cur_overcom, cur_incom, cur_shift, cur_fp, cur_tp_iou, cur_overcom_iou, cur_incom_iou, cur_shift_iou =  \
            check_completeness(pred_of_cur_cls, gt_of_cur_cls, matched_result, matched_iou)
            
            if cur_tp_iou != 0:
                tp_iou.append(cur_tp_iou)
            if cur_overcom_iou != 0:
                overcom_iou.append(cur_overcom_iou)
            if cur_incom_iou != 0:
                incom_iou.append(cur_incom_iou)
            if cur_shift_iou != 0:
                shift_iou.append(cur_shift_iou)
            
            tp += cur_tp
            overcom += cur_overcom
            incom += cur_incom
            shift += cur_shift
            fp += cur_fp
            total += cur_tp + cur_overcom + cur_incom + cur_shift + cur_fp
            
        if true_vid_total_pred != total: # 비디오에 존재하지 않는 클래스를 모델이 예측한 경우 -> fp로 추가
            fp += true_vid_total_pred - total
            total += true_vid_total_pred - total
            
    print(f'Total prediction: {cal_pred.shape[0]} \n TP: {tp} \n Overcomplete: {overcom} \n \
    Incomplete: {incom} \n Shifted: {shift} \n FP: {fp} \n sum: {tp+overcom+incom+shift+fp}')

    print(np.mean(np.array(tp_iou)))
    print(np.mean(np.array(overcom_iou)))
    print(np.mean(np.array(incom_iou)))
    print(np.mean(np.array(shift_iou)))