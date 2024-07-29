import torch
import numpy as np

import model.clip as clip

from collections import OrderedDict

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id

def classes(cls):
    return cls.lower()

def text_prompt(dataset='Thumos14reduced', clipbackbone='R50', device='cpu'):
    actionlist, actionprompt, actiontoken = [], {}, []
    numC = {'Thumos14reduced': 20, 'ActivityNet1.2':100, 'ActivityNet1.3':200}
    
    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False


    meta = np.load("../../data/annet/ActivityNet1.3/ActivityNet1.3-Annotations/classlist.npy", 'r')
    actionlist = [classes(act) for act in meta]

    actionlist = np.array([a.split('\n')[0] for a in actionlist])
    actiontoken = np.array([convert_to_token(a) for a in actionlist])
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device)) # [20, 1, 77, 512]

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC[dataset]))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC[dataset]))

    return actionlist, actiondict, actiontoken
