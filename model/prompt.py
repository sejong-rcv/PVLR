import torch
import numpy as np

import model.clip as clip

from collections import OrderedDict

def convert_to_token(xh):
    xh_id = clip.tokenize(xh).cpu().data.numpy()
    return xh_id

classes = {
            'BaseballPitch': 'baseball pitch',
            'BasketballDunk': 'basketball dunk',
            'Billiards': 'billiards',
            'CleanAndJerk': 'clean and jerk',
            'CliffDiving': 'cliff diving',
            'CricketBowling': 'cricket bowling',
            'CricketShot': 'cricket shot',
            'Diving': 'diving',
            'FrisbeeCatch': 'frisbee catch',
            'GolfSwing': 'golf swing',
            'HammerThrow': 'hammer throw',
            'HighJump': 'high jump',
            'JavelinThrow': 'javelin throw',
            'LongJump': 'long jump',
            'PoleVault': 'pole vault',
            'Shotput': 'shot put',
            'SoccerPenalty': 'soccer penalty',
            'TennisSwing': 'tennis swing',
            'ThrowDiscus': 'throw discus',
            'VolleyballSpiking': 'volleyball spiking'
}

def text_prompt(dataset='Thumos14reduced', clipbackbone='R50', device='cpu'):
    actionlist, actionprompt, actiontoken = [], {}, []
    numC = {'Thumos14reduced': 20,}
    
    # load the CLIP model
    clipmodel, _ = clip.load(clipbackbone, device=device, jit=False)
    for paramclip in clipmodel.parameters():
        paramclip.requires_grad = False

    # convert to token, will automatically padded to 77 with zeros
    if dataset == 'Thumos14reduced':
        meta = np.load("features/Thumos14reduced-Annotations/classlist.npy", 'r')
        actionlist = [classes[act.decode('utf-8')] for act in meta]
        # actionlist = meta.readlines()
        # meta.close()
        actionlist = np.array([a.split('\n')[0] for a in actionlist])
        actiontoken = np.array([convert_to_token(a) for a in actionlist])

    # More datasets to be continued
    # query the vector from dictionary
    with torch.no_grad():
        actionembed = clipmodel.encode_text_light(torch.tensor(actiontoken).to(device)) # [20, 1, 77, 512]

    actiondict = OrderedDict((actionlist[i], actionembed[i].cpu().data.numpy()) for i in range(numC[dataset]))
    actiontoken = OrderedDict((actionlist[i], actiontoken[i]) for i in range(numC[dataset]))

    return actionlist, actiondict, actiontoken
