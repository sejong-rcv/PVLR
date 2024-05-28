import numpy as np
import torch
import utils.wsad_utils as utils
import pandas as pd
import options
args = options.parser.parse_args()

def get_cls_score(element_cls, dim=-2, rat=20, ind=None):
    topk_val, _ = torch.topk(element_cls,
                             k=max(1, int(element_cls.shape[-2] // rat)),
                             dim=-2)
    instance_logits = torch.mean(topk_val, dim=-2)
    pred_vid_score = torch.softmax(
        instance_logits, dim=-1)[..., :-1].squeeze().data.cpu().numpy()
    return pred_vid_score

@torch.no_grad()
def multiple_threshold_hamnet(vid_name,data_dict,args):
    elem = data_dict['cas']
    element_atn=data_dict['attn']
    act_thresh_cas = np.arange(0.1, 0.9, 10)
    element_logits = elem * element_atn
   
    pred_vid_score = get_cls_score(element_logits, rat=10)
    score_np = pred_vid_score.copy()
 
    cas_supp = element_logits[..., :-1]
    cas_supp_atn = element_atn

    pred = np.where(pred_vid_score >= args.vid_threshold)[0]

    act_thresh = np.linspace(args.act_s,args.act_e,args.act_num)
    cas_thresh = np.linspace(args.cas_s, args.cas_e, args.cas_num)
    prediction = None

    if len(pred) == 0:
        pred = np.array([np.argmax(pred_vid_score)])
    
    cas_pred = cas_supp[0].cpu().numpy()[:, pred]
    num_segments = cas_pred.shape[0]
    cas_pred = np.reshape(cas_pred, (num_segments, -1, 1))

    cas_pred_minmax = (cas_pred - cas_pred.min(axis=0)) / (cas_pred.max(axis=0)-cas_pred.min(axis=0))

    cas_pred_atn = cas_supp_atn[0].cpu().numpy()[:, [0]]
    cas_pred_atn = np.reshape(cas_pred_atn, (num_segments, -1, 1))
    proposal_dict = {}

    # CAS
    for i in range(len(cas_thresh)):
        cas_temp = cas_pred_minmax.copy()
        seg_list = []

        for c in range(len(pred)):
            pos = np.where(cas_temp[:, c, 0] > cas_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                            cas_temp,
                                            pred_vid_score,
                                            pred,
                                            args.cas_scale,
                                            num_segments,
                                            args.feature_fps,
                                            num_segments,
                                            gamma=args.cas_gamma_oic,
                                            lambda_=args.cas_lambda_)
        
        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                print('index error')

    # Aness
    for i in range(len(act_thresh)):
        cas_temp = cas_pred.copy()
        cas_temp_atn = cas_pred_atn.copy()
        seg_list = []
        for c in range(len(pred)):
            pos = np.where(cas_temp_atn[:, 0, 0] > act_thresh[i])
            seg_list.append(pos)

        proposals = utils.get_proposal_oic_2(seg_list,
                                            cas_temp,
                                            pred_vid_score,
                                            pred,
                                            args.act_scale,
                                            num_segments,
                                            args.feature_fps,
                                            num_segments,
                                            gamma=args.act_gamma_oic,
                                            lambda_=args.act_lambda_)

        for j in range(len(proposals)):
            try:
                class_id = proposals[j][0][0]

                if class_id not in proposal_dict.keys():
                    proposal_dict[class_id] = []

                proposal_dict[class_id] += proposals[j]
            except IndexError:
                logger.error(f"Index error")
    final_proposals = []
    for class_id in proposal_dict.keys():
        final_proposals.append(
            utils.soft_nms(proposal_dict[class_id], args.soft_nms_thresh, sigma=args.sigma))

    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    segment_predict = []
    for i in range(len(final_proposals)):
        for j in range(len(final_proposals[i])):
            [c_pred, c_score, t_start, t_end] = final_proposals[i][j]
            segment_predict.append([t_start, t_end,c_score,c_pred])

    segment_predict = np.array(segment_predict)
    
    video_lst, t_start_lst, t_end_lst = [], [], []
    label_lst, score_lst = [], []
    
    for i in range(np.shape(segment_predict)[0]):
        video_lst.append(vid_name)
        t_start_lst.append(segment_predict[i, 0])
        t_end_lst.append(segment_predict[i, 1])
        score_lst.append(segment_predict[i, 2])
        label_lst.append(segment_predict[i, 3])
    prediction = pd.DataFrame(
        {
            "video-id": video_lst,
            "t-start": t_start_lst,
            "t-end": t_end_lst,
            "label": label_lst,
            "score": score_lst,
        }
    )
    return prediction

