import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init

import utils.wsad_utils as utils
from torch.nn import init

import model.clip as clip
from model.prob_encoder import SnippetEncoder

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)

class BWA_fusion_dropout_feat_v2(torch.nn.Module):
    def __init__(self, n_feature, n_class,**args):
        super().__init__()
        embed_dim = 1024
        self.bit_wise_attn = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.channel_conv = nn.Sequential(
            nn.Conv1d(n_feature, embed_dim, 3, padding=1),nn.LeakyReLU(0.2),nn.Dropout(0.5))
        self.attention = nn.Sequential(nn.Conv1d(embed_dim, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2),
                                       nn.Dropout(0.5),
                                       nn.Conv1d(512, 512, 3, padding=1),
                                       nn.LeakyReLU(0.2), nn.Conv1d(512, 1, 1),
                                       nn.Dropout(0.5),
                                       nn.Sigmoid())
        self.channel_avg=nn.AdaptiveAvgPool1d(1)

    def forward(self,vfeat,ffeat):
        channelfeat = self.channel_avg(vfeat)
        channel_attn = self.channel_conv(channelfeat)
        bit_wise_attn = self.bit_wise_attn(ffeat)
        filter_feat = torch.sigmoid(bit_wise_attn*channel_attn)*vfeat
        x_atn = self.attention(filter_feat)
        return x_atn, filter_feat

class Similarity(nn.Module):
    def __init__(self, train_sig_T, test_sig_T):
        super().__init__()
        self.train_sig_T = train_sig_T
        self.test_sig_T = test_sig_T

    def forward(self, v_feat, t_feat, is_training):
        b, n, t, d = v_feat.shape
        t_feat = t_feat.unsqueeze(0).unsqueeze(0).repeat(b, n, 1, 1)

        if is_training:
            tau = self.train_sig_T
        if not is_training:
            tau = self.test_sig_T
            v_feat = torch.nn.functional.normalize(v_feat, dim=-1)
            t_feat = torch.nn.functional.normalize(t_feat, dim=-1)

        dist = torch.einsum('bntd,bmcd->bctnm',[v_feat,t_feat]) / tau
        dist = torch.mean(torch.mean(dist,dim=-1),dim=-1)
        return dist

class PECR(torch.nn.Module):
    def __init__(self, n_feature, n_class, actiondict, actiontoken, inp_actionlist, **args):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.actiondict = actiondict
        self.actiontoken = actiontoken
        self.inp_actionlist = inp_actionlist
        
        embed_dim=2048
        dropout_ratio=args['opt'].dropout_ratio

        self.P_v = args['opt'].num_prob_v
        
        self.prefix = args['opt'].prefix
        self.postfix = args['opt'].postfix

        self.train_sig_T = args['opt'].train_sig_T
        self.test_sig_T = args['opt'].test_sig_T
        self.convex_alpha = args['opt'].convex_alpha

        self.vAttn = BWA_fusion_dropout_feat_v2(1024, args)
        self.fAttn = BWA_fusion_dropout_feat_v2(1024, args)

        self.fusion = nn.Sequential(
            nn.Conv1d(n_feature, n_feature, 1, padding=0),
            nn.LeakyReLU(0.2),nn.Dropout(dropout_ratio))
        self.fusion2 = nn.Sequential(
            nn.Conv1d(embed_dim,embed_dim, 3,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_ratio)
        )

        _kernel = ((args['opt'].max_seqlen // args['opt'].t) // 2 * 2 + 1)
        self.pool=nn.AvgPool1d(_kernel, 1, padding=_kernel // 2, count_include_pad=True) \
            if _kernel is not None else nn.Identity()
        self.apply(weights_init)

        self.snippet_prob_encoder = SnippetEncoder(args['opt'].eps_std, 2048)
        self.matching_prob = Similarity(train_sig_T=self.train_sig_T, test_sig_T=self.test_sig_T) 

        self.hidden_size = 512
        self.embedding = torch.nn.Embedding(77, self.hidden_size)
        self.initialize_parameters()

        self.clipmodel, _ = clip.load(args['opt'].backbone, device=self.device, jit=False) 
        self.clipmodel.eval()
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

    def forward(self, inputs, clip_feat, is_training=True, **args):
        feat = inputs.transpose(-1, -2)
        b,c,n=feat.size()

        v_atn,vfeat = self.vAttn(feat[:,:1024,:],feat[:,1024:,:])
        f_atn,ffeat = self.fAttn(feat[:,1024:,:],feat[:,:1024,:])

        x_atn = self.convex_alpha*f_atn + (1-self.convex_alpha)*v_atn
        nfeat = torch.cat((vfeat,ffeat),1)

        feat_loss = self.fusion(nfeat) 
        feat_cas = self.fusion2(feat_loss)

        self.replace_text_embedding(self.inp_actionlist)
        text_feature = self.clipmodel.encode_text(self.text_embedding, self.prompt_actiontoken)
        text_feature = text_feature.to(torch.float32) 

        mu_v, emb_v, var_v = self.snippet_prob_encoder(feat_cas, self.P_v, is_training) 
        cas = self.matching_prob(emb_v, text_feature, is_training) 

        x_atn=self.pool(x_atn)
        f_atn=self.pool(f_atn)
        v_atn=self.pool(v_atn)

        return {'feat':feat_loss.transpose(-1, -2), 'cas':cas.transpose(-1, -2), 'attn':x_atn.transpose(-1, -2), 'v_atn':v_atn.transpose(-1, -2),'f_atn':f_atn.transpose(-1, -2),
                'mu_v': mu_v, 'var_v': var_v, 'emb_v': emb_v, 'text_feat':text_feature}