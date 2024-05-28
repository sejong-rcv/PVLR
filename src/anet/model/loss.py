import torch
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
from scipy import ndimage

class ProbLoss(torch.nn.Module):
    def __init__(self, args):
        super(ProbLoss, self).__init__()
        self.args = args
        self.ce_criterion = torch.nn.CrossEntropyLoss()
        self.dropout = torch.nn.Dropout(p=0.6)

        self.k_easy = args.k_easy
        self.k_hard = args.k_hard

        self.M = args.M
        self.m = args.m

    def select_topk_embeddings(self, scores, embeddings, k):
        _, idx_DESC = scores.sort(descending=True, dim=1)
        idx_topk = idx_DESC[:, :k]
        idx_topk = idx_topk.unsqueeze(2).expand([-1, -1, embeddings.shape[2]])
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)
        return selected_embeddings

    def easy_snippets_mining(self, actionness, mu, var):
        actionness = actionness.squeeze()
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)

        actionness_drop = actionness * select_idx

        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx

        easy_act_mu = self.select_topk_embeddings(actionness_drop, mu, k=self.k_easy)
        easy_act_var = self.select_topk_embeddings(actionness_drop, var, k=self.k_easy)

        easy_bkg_mu = self.select_topk_embeddings(actionness_rev_drop, mu, k=self.k_easy)
        easy_bkg_var = self.select_topk_embeddings(actionness_rev_drop, var, k=self.k_easy)

        k=max(1, int(mu.shape[-2] // self.k_easy))
        return (easy_act_mu, easy_act_var), (easy_bkg_mu, easy_bkg_var)

    def hard_snippets_mining(self, actionness, mu, var):
        actionness = actionness.squeeze()

        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        erosion_M = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        erosion_m = ndimage.binary_erosion(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        idx_region_inner = actionness.new_tensor(erosion_m - erosion_M)
        aness_region_inner = actionness * idx_region_inner
        hard_act_mu = self.select_topk_embeddings(aness_region_inner, mu, k=self.k_hard)
        hard_act_var = self.select_topk_embeddings(aness_region_inner, var, k=self.k_hard)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg_mu = self.select_topk_embeddings(aness_region_outer, mu, k=self.k_hard)
        hard_bkg_var = self.select_topk_embeddings(aness_region_outer, var, k=self.k_hard)

        return (hard_act_mu, hard_act_var), (hard_bkg_mu, hard_bkg_var)

    def KL_divergence(self, mu_p, cov_p, mu_q, cov_q):
        distance = []
        cov_p = cov_p + 1e-5
        cov_q = cov_q + 1e-5
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                term1 = 0.5*torch.einsum('bd,bd,bd->b',[(mu_q[:,j,:]-mu_p[:,i,:]), 1/cov_q[:,j,:], (mu_q[:,j,:]-mu_p[:,i,:])])
                term2 = 0.5*(torch.log(cov_q[:,j,:]).sum(-1) - torch.log(cov_p[:,i,:]).sum(-1))
                term3 = 0.5*((cov_p[:,i,:]/cov_q[:,j,:]).sum(-1))
                dist = term1 + term2 + term3 - 0.5*mu_p.shape[2]
                distance.append(1/(dist+1))
        distance = torch.stack(distance,dim=-1)
        return distance.mean(-1)

    def Bhattacharyya_distance(self, mu_p, cov_p, mu_q, cov_q): 
        distance = []
        cov_p = cov_p + 1e-5
        cov_q = cov_q + 1e-5
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                term1 = 0.125*torch.einsum('bd,bd,bd->b',[(mu_p[:,i,:]-mu_q[:,j,:]), 2/(cov_p[:,i,:]+cov_q[:,j,:]), (mu_p[:,i,:]-mu_q[:,j,:])])
                term2 = 0.5*(torch.log((cov_p[:,i,:]+cov_q[:,j,:])/2).sum(-1)-(torch.log(cov_p[:,i,:]).sum(-1)+torch.log(cov_q[:,j,:]).sum(-1)))
                dist = term1 + term2
                distance.append(1/(dist+1))
        distance = torch.stack(distance,dim=-1)
        return distance.mean(-1)

    def Mahalanobis_distance(self, mu_p, cov_p, mu_q, cov_q):
        distance = []
        for i in range(mu_p.shape[1]):
            for j in range(mu_q.shape[1]):
                cov_inv = 2/(cov_p[:,i,:]+cov_q[:,j,:]+1e-5)
                dist = torch.einsum('bd,bd,bd->b',[(mu_p[:,i,:]-mu_q[:,j,:]), cov_inv, (mu_p[:,i,:]-mu_q[:,j,:])])
                distance.append(1/(dist+1))
        distance = torch.stack(distance, dim=-1)
        return distance.mean(-1)

    def NCE(self, q, k, neg, T=0.07):
        q = torch.nn.functional.normalize(q, dim=1)
        k = torch.nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = torch.nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def ProbabilsticContrastive(self, hard_query, easy_pos, easy_neg):
        if self.args.metric == 'Mahala':
            pos_distance = self.Mahalanobis_distance(hard_query[0],hard_query[1],easy_pos[0],easy_pos[1])
            neg_distance = self.Mahalanobis_distance(hard_query[0],hard_query[1],easy_neg[0],easy_neg[1])

        elif self.args.metric == 'KL_div':
            pos_distance = self.KL_divergence(hard_query[0],hard_query[1],easy_pos[0],easy_pos[1])
            neg_distance = self.KL_divergence(hard_query[0],hard_query[1],easy_neg[0],easy_neg[1])

        elif self.args.metric == 'Bhatta':
            pos_distance = self.Bhattacharyya_distance(hard_query[0],hard_query[1],easy_pos[0],easy_pos[1])
            neg_distance = self.Bhattacharyya_distance(hard_query[0],hard_query[1],easy_neg[0],easy_neg[1])

        loss = -1*(torch.log(pos_distance) + torch.log(1-neg_distance))

        return loss.mean()

    def Distillation(self, mu, clip_feat):
        mu = torch.nn.functional.normalize(mu,dim=-1)
        clip_feat = torch.nn.functional.normalize(clip_feat,dim=-1)
        sim = torch.einsum('btd,btd->bt',[mu,clip_feat])
        sim = (sim + 1) / 2
        return -torch.log(sim.mean())
    
    def orthogonalization(self, emb):
        emb = torch.nn.functional.normalize(emb,dim=-1)
        sim_matrix = emb @ emb.T.detach()
        sim_matrix = sim_matrix - torch.eye(len(sim_matrix), device=sim_matrix.device)
        return sim_matrix.norm()
    
    def forward(self, iter, data, mu_clip, labels):

        attn, mu, var, category_emb = data['attn'], data['mu_v'], data['var_v'], data['text_feat']

        easy_act, easy_bkg = self.easy_snippets_mining(attn, mu, var)
        hard_act, hard_bkg = self.hard_snippets_mining(attn, mu, var)
        
        distillation_loss = self.args.alpha4 * self.Distillation(mu, mu_clip)
        action_prob_contra_loss = self.args.alpha5 * self.ProbabilsticContrastive(hard_act, easy_act, easy_bkg)
        background_prob_contra_loss = self.args.alpha6 * self.ProbabilsticContrastive(hard_bkg, easy_bkg, easy_act)
        ortho_loss = self.args.alpha7 * self.orthogonalization(category_emb)

        return distillation_loss + action_prob_contra_loss + background_prob_contra_loss + ortho_loss, \
               (distillation_loss, action_prob_contra_loss, background_prob_contra_loss, ortho_loss)

class VideoLoss(torch.nn.Module):
    def __init__(self, args):
        super(VideoLoss, self).__init__()
        self.args = args

    def _multiply(self, x, atn, dim=-1, include_min=False):
        if include_min:
            _min = x.min(dim=dim, keepdim=True)[0]
        else:
            _min = 0
        return atn * (x - _min) + _min

    def topkloss(self, element_logits, labels, is_back=True, lab_rand=None, rat=8, reduce=None):
        if is_back:
            labels_with_back = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels_with_back = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
        if lab_rand is not None:
            labels_with_back = torch.cat((labels, lab_rand), dim=-1)
        
        topk_val, topk_ind = torch.topk(
            element_logits, 
            k=max(1, int(element_logits.shape[-2] // rat)), 
            dim=-2) 
        instance_logits = torch.mean(
            topk_val, 
            dim=-2)

        labels_with_back = labels_with_back / (
            torch.sum(labels_with_back, dim=1, keepdim=True) + 1e-4)
    
        milloss = (-(labels_with_back * 
                     F.log_softmax(instance_logits, dim=-1)).sum(dim=-1))
        if reduce is not None:
            milloss = milloss.mean()
        return milloss, topk_ind

    def Contrastive(self, x, element_logits, labels, is_back=False):
        # background class
        if is_back:
            labels = torch.cat(
                (labels, torch.ones_like(labels[:, [0]])), dim=-1)
        else:
            labels = torch.cat(
                (labels, torch.zeros_like(labels[:, [0]])), dim=-1)
            
        sim_loss = 0.
        n_tmp = 0.
        _, n, c = element_logits.shape
        for i in range(0, 3*2, 2):
            atn1 = F.softmax(element_logits[i], dim=0)
            atn2 = F.softmax(element_logits[i+1], dim=0)

            n1 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            n2 = torch.FloatTensor([np.maximum(n-1, 1)]).cuda()
            Hf1 = torch.mm(torch.transpose(x[i], 1, 0), atn1)
            Hf2 = torch.mm(torch.transpose(x[i+1], 1, 0), atn2)
            Lf1 = torch.mm(torch.transpose(x[i], 1, 0), (1 - atn1)/n1)
            Lf2 = torch.mm(torch.transpose(x[i+1], 1, 0), (1 - atn2)/n2)

            d1 = 1 - torch.sum(Hf1*Hf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Hf2, 2, dim=0))
            d2 = 1 - torch.sum(Hf1*Lf2, dim=0) / (torch.norm(Hf1, 2, dim=0) * torch.norm(Lf2, 2, dim=0))
            d3 = 1 - torch.sum(Hf2*Lf1, dim=0) / (torch.norm(Hf2, 2, dim=0) * torch.norm(Lf1, 2, dim=0))
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d2+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            sim_loss = sim_loss + 0.5*torch.sum(torch.max(d1-d3+0.5, torch.FloatTensor([0.]).cuda())*labels[i,:]*labels[i+1,:])
            n_tmp = n_tmp + torch.sum(labels[i,:]*labels[i+1,:])
        sim_loss = sim_loss / n_tmp
        return sim_loss

    def forward(self, data, labels):
        feat, element_logits, element_atn, v_atn, f_atn = data['feat'], data['cas'], data['attn'], data['v_atn'], data['f_atn']
        mutual_loss = 0

        b,n,c = element_logits.shape
        element_logits_supp = self._multiply(element_logits, element_atn, include_min=True)

        # classification loss
        loss_mil_orig, _ = self.topkloss(element_logits, labels, is_back=True, rat=self.args.train_topk, reduce=None)
        loss_mil_supp, _ = self.topkloss(element_logits_supp, labels, is_back=False, rat=self.args.train_topk, reduce=None)

        # contrastive loss
        loss_supp_Contrastive = self.Contrastive(feat, element_logits_supp, labels, is_back=False)

        # normalization loss
        loss_norm = element_atn.mean()
        v_loss_norm = v_atn.mean()
        f_loss_norm = f_atn.mean()

        # guide loss
        loss_guide = (1 - element_atn -element_logits.softmax(-1)[..., [-1]]).abs().mean()
        v_loss_guide = (1 - v_atn -element_logits.softmax(-1)[..., [-1]]).abs().mean()
        f_loss_guide = (1 - f_atn -element_logits.softmax(-1)[..., [-1]]).abs().mean() 

        # video loss
        cls_loss = (loss_mil_orig.mean() + loss_mil_supp.mean())
        norm_loss = self.args.alpha1*(loss_norm + v_loss_norm + f_loss_norm)/3
        guide_loss = self.args.alpha2*(loss_guide + v_loss_guide + f_loss_guide)/3
        contra_loss = self.args.alpha3*loss_supp_Contrastive

        return cls_loss + norm_loss + guide_loss + contra_loss + mutual_loss, (cls_loss, norm_loss, guide_loss, contra_loss, mutual_loss)

class TotalLoss(torch.nn.Module):
    def __init__(self, args):
        super(TotalLoss, self).__init__()
        self.args = args
        self.video_criterion = VideoLoss(args)   
        self.prob_criterion = ProbLoss(args)   

    def forward(self, iter, outputs, clip_feature, labels):
        video_loss, (cls_loss, norm_loss, guide_loss, contra_loss, mutual_loss) = self.video_criterion(outputs, labels)
        prob_loss, (distillation_loss, action_prob_contra_loss, background_prob_contra_loss, ortho_loss) = self.prob_criterion(iter, outputs, clip_feature, labels)

        return video_loss + prob_loss, {"cls_loss":cls_loss, "norm_loss":norm_loss, "guide_loss":guide_loss,
                                         "contra_loss":contra_loss, "distillation_loss":distillation_loss,
                                          "action_prob_contra_loss":action_prob_contra_loss, 
                                          "background_prob_contra_loss":background_prob_contra_loss, 
                                          "ortho_loss":ortho_loss, "mutual_loss": mutual_loss}
