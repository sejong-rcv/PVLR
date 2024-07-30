import nltk
import numpy as np
import torch
import torch.nn as nn
import model.clip as clip

from model.prob_encoder import SnippetEncoder

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class Attn(torch.nn.Module):
    def __init__(self, n_feature, temperature):
        super().__init__()
        embed_dim = 1024
        self.tau = temperature

        self.AE_e = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim//2, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5)) 
        
        self.AE_d = nn.Sequential(
            nn.Conv1d(embed_dim//2, n_feature, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))

        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature//2, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),nn.LeakyReLU(0.2), nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1), nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1), nn.Dropout(0.5))
        
        self.channel_avg=nn.AdaptiveAvgPool1d(1)

    def forward(self,vfeat,ffeat): 
        fusion_feat = self.AE_e(ffeat)
        new_feat = self.AE_d(fusion_feat)

        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat) 
        channel_attn_norm = channel_attn/torch.norm(channel_attn,p=2,dim=1,keepdim=True)

        bit_wise_attn = self.bit_wise_attn(fusion_feat) # b, 1024, 320
        bit_wise_attn_norm = bit_wise_attn/torch.norm(bit_wise_attn,p=2,dim=1,keepdim=True)

        temp_attn= torch.einsum('bdn,bdt->bnt',[channel_attn_norm, bit_wise_attn_norm]) # [10, 1, 320]

        filter_feat = torch.sigmoid(bit_wise_attn*temp_attn)*vfeat
        x_atn = self.attention(filter_feat)
        x_atn = torch.sigmoid(x_atn/self.tau)
        return x_atn, filter_feat, new_feat, vfeat

class Similarity(nn.Module):
    def __init__(self, sig_T_train, sig_T_infer):
        super().__init__()
        self.sig_T_train = sig_T_train
        self.sig_T_infer = sig_T_infer

    def forward(self, v_feat, t_feat, split): 
        
        b, n, t, d = v_feat.shape
        t_feat = t_feat.unsqueeze(0).unsqueeze(0).repeat(b, n, 1, 1)

        if split == 'test':
            tau = self.sig_T_infer
            v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
            t_feat = torch.nn.functional.normalize(t_feat, dim=-1)
        else:
            tau = self.sig_T_train

        dist = torch.einsum('bntd,bmcd->bctnm',[v_feat,t_feat]) / tau
        dist = torch.mean(torch.mean(dist,dim=-1),dim=-1)
        return dist

class PVLR(torch.nn.Module):
    def __init__(self, actiondict, actiontoken, inp_actionlist,**args):
        super().__init__()
        embed_dim=2048
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.vAttn = Attn(1024, args['opt'].sig_T_attn)
        self.fAttn = Attn(1024, args['opt'].sig_T_attn)

        self.fusion = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7)
            )
        
        self.fusion2 = nn.Sequential(
            nn.Conv1d(embed_dim,embed_dim, 3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.7)
            )

        self.P_v = args['opt'].num_prob_v
        self.n_class = args['opt'].num_class

        self.sig_T_train = args['opt'].sig_T_train
        self.sig_T_infer = args['opt'].sig_T_infer

        self.prefix = args['opt'].prefix
        self.postfix = args['opt'].postfix
      
        self.snippet_prob_encoder = SnippetEncoder(dim=2048,eps_std=args['opt'].eps_std)

        self.matching_prob = Similarity(sig_T_train=self.sig_T_train, sig_T_infer=self.sig_T_infer) 
        self.clipmodel, _ = clip.load(args['opt'].backbone, device=self.device, jit=False) 
        
        self.hidden_size = 512
        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.inp_actionlist = inp_actionlist

        self.clipmodel.eval()

        self.initialize_parameters()

        for paramclip in self.clipmodel.parameters():
            paramclip.requires_grad = False

    def initialize_parameters(self):
        nn.init.normal_(self.embedding.weight, std=0.01)

    def replace_text_embedding(self, actionlist):
        self.text_embedding = self.embedding(torch.arange(77).to(self.device))[None, :].repeat([len(actionlist)+1, 1, 1]) 
        self.prompt_actiontoken = torch.zeros(len(actionlist)+1, 77) 
        for i, a in enumerate(actionlist):

            embedding = torch.from_numpy(self.actiondict[a][0]).float().to(self.device) 
            token = torch.from_numpy(self.actiontoken[a][0])

            self.text_embedding[i][0] = embedding[0] 
            ind = np.argmax(token, -1)

            self.text_embedding[i][self.prefix + 1: self.prefix + ind] = embedding[1:ind]
            self.text_embedding[i][self.prefix + ind + self.postfix] = embedding[ind]

            self.prompt_actiontoken[i][0] = token[0]
            self.prompt_actiontoken[i][self.prefix + 1: self.prefix + ind] = token[1:ind]
            self.prompt_actiontoken[i][self.prefix + ind + self.postfix] = token[ind]

        self.text_embedding.to(self.device)
        self.prompt_actiontoken.to(self.device)

    def forward(self, inputs, clip_feat, itr, split, **args):
        feat = inputs.transpose(-1, -2)

        v_atn, vfeat, n_rfeat, o_rfeat = self.vAttn(feat[:,:1024,:],feat[:,1024:,:])
        f_atn, ffeat, n_ffeat, o_ffeat = self.fAttn(feat[:,1024:,:],feat[:,:1024,:])

        x_atn = args['opt'].convex_alpha*f_atn+(1-args['opt'].convex_alpha)*v_atn

        nfeat = torch.cat((vfeat,ffeat),1)
        
        nfeat_out0 = self.fusion(nfeat)
        nfeat_out = self.fusion2(nfeat_out0) 

        # Text embedding
        self.replace_text_embedding(self.inp_actionlist)

        text_feature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken) 
        text_feature = text_feature.to(torch.float32) 

        ################# Probabilistic CAS #################
        mu_v, emb_v, var_v = self.snippet_prob_encoder(itr, nfeat_out, self.P_v, split) 
        cas = self.matching_prob(emb_v, text_feature, split)
    
        with torch.no_grad():
            reparm_sim = torch.einsum("bktd,btd->bkt",[torch.nn.functional.normalize(emb_v ,dim=-1), torch.nn.functional.normalize(mu_v, dim=-1)]).mean(dim=1)
            distill_sim = torch.einsum("btd,btd->bt",[torch.nn.functional.normalize(mu_v ,dim=-1), torch.nn.functional.normalize(clip_feat, dim=-1)])

        return {'feat':nfeat_out0.transpose(-1, -2),'text_feat':text_feature,'feat_final':nfeat_out.transpose(-1,-2), 'cas':cas.transpose(-1,-2), 'attn':x_atn.transpose(-1,-2), \
                'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),'n_rfeat':n_rfeat.transpose(-1,-2),'o_rfeat':o_rfeat.transpose(-1,-2), \
                'n_ffeat':n_ffeat.transpose(-1,-2),'o_ffeat':o_ffeat.transpose(-1,-2), 'mu_v':mu_v, 'var_v':var_v, 'emb_v':emb_v, "reparm_sim":reparm_sim, "distill_sim":distill_sim
                }