from __future__ import print_function
import os
import json
import numpy as np
import utils.wsad_utils as utils

def get_video_prompt_templates():
    prompts = [
        'one video of the ',    
    ]
    return prompts

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

class SampleDataset:
    def __init__(self, args, mode="both", sampling='random'):

        self.dataset_name = args.dataset_name
        self.num_class = args.num_class
        self.sampling=sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size
        self.path_to_features = os.path.join("features/Thumos14reduced/Thumos14reduced-I3D-JOINTFeatures.npy")
        self.path_to_annotations = os.path.join("features/Thumos14reduced-Annotations/")
        self.features = np.load(
            self.path_to_features, encoding="bytes", allow_pickle=True
        )
        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
        # Specific to Thumos14
        self._labels = np.load(
            self.path_to_annotations + "labels.npy", allow_pickle=True
        )
        self.classlist = np.load(
            self.path_to_annotations + "classlist.npy", allow_pickle=True
        )
        self.subset = np.load(
            self.path_to_annotations + "subset.npy", allow_pickle=True
        )
        self.videonames = np.load(
            self.path_to_annotations + "videoname.npy", allow_pickle=True
        )
        self.duration = np.load(
            self.path_to_annotations + "duration.npy", allow_pickle=True
        )
        self.fps = np.load(
            self.path_to_annotations + "original_fps.npy", allow_pickle=True
        )    

        self.clip_path = args.path_clip_dataset
        self.batch_size = args.batch_size
        self.len_txt = 20
        self.trainidx = []
        self.testidx = []
        self.classwiseidx = []
        self.currenttestidx = 0

        self.labels_multihot = [
            utils.strlist2multihot(labs, self.classlist)
            for labs in self.labels
        ]

        try:
            ambilist = self.path_to_annotations + "/Ambiguous_test.txt"
            ambilist = list(open(ambilist, "r"))
            ambilist = [a.strip("\n").split(" ")[0] for a in ambilist]
        except:
            ambilist = []

        self.train_test_idx()
        self.classwise_feature_mapping()

        self.normalize = False
        self.mode = mode
        if mode == "rgb" or mode == "flow":
            self.feature_size = 1024

        with open('features/Thumos14reduced/gt.json') as j:
            self.anno = json.load(j)
        self.temp_label = self.get_temp_label()
        
    def get_temp_label(self): 
        feature_len = []
        temp_anno = []
        label_dict = {v.decode('utf-8'):k for k, v in enumerate(self.classlist)}

        for feat in self.features:
            feature_len.append(feat.shape[0])

        for i, (gt, dur) in enumerate(zip(self.segments, self.duration)): # video
            cur_feature_len = feature_len[i]
            cur_label = np.zeros((self.num_class, cur_feature_len))
            cur_duration = self.duration[i]
            for idx, anno in enumerate(gt): # gt
                cur_class = label_dict[self._labels[i][idx]]
                start = int(np.round(anno[0]*cur_feature_len/cur_duration))
                end = int(np.round(anno[1]*cur_feature_len/cur_duration))
                cur_label[cur_class, start:end+1] = 1
            temp_anno.append(cur_label)
        return temp_anno
    
    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s.decode("utf-8") == "validation":  # Specific to Thumos14
                self.trainidx.append(i)
            elif s.decode("utf-8") == "test":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category.decode("utf-8"):
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, n_pro=14, n_similar=0, is_training='train', similar_size=2):
        if is_training == 'train':
            labels = []
            idx = []

            # Load similar pairs
            if n_similar != 0:
                rand_classid = np.random.choice(
                    len(self.classwiseidx), size=n_similar
                )
                for rid in rand_classid:
                    rand_sampleid = np.random.choice(
                        len(self.classwiseidx[rid]),
                        size=similar_size,
                        replace=False,
                    )

                    for k in rand_sampleid:
                        idx.append(self.classwiseidx[rid][k])

            # Load rest pairs
            if self.batch_size - similar_size * n_similar < 0:
                self.batch_size = similar_size * n_similar

            rand_sampleid = np.random.choice(
                len(self.trainidx),
                size=self.batch_size - similar_size * n_similar,
            )

            for r in rand_sampleid:
                idx.append(self.trainidx[r])
                
            feat = []
            temp_anno = []
            clip_feat = []

            for i in idx:
                ifeat = self.features[i]
                labs = self.labels[i]
                clip_ifeat = np.load(os.path.join(self.clip_path, 'train', self.videonames[i].decode('utf-8')+'.npy'))
                if self.sampling == 'random':
                    sample_idx = self.random_perturb(ifeat.shape[0])
                elif self.sampling == 'uniform':
                    sample_idx = self.uniform_sampling(ifeat.shape[0])
                elif self.sampling == "all":
                    sample_idx = np.arange(ifeat.shape[0])
                else:
                    raise AssertionError('Not supported sampling !')
                ifeat = ifeat[sample_idx]
                clip_ifeat = clip_ifeat[sample_idx]
                feat.append(ifeat)
                clip_feat.append(clip_ifeat)
                cur_temp_anno = self.temp_label[i][:, sample_idx]
                temp_anno.append(cur_temp_anno)
            
            feat = np.array(feat)
            clip_feat = np.array(clip_feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]

            return feat, clip_feat, np.array(temp_anno), labels, rand_sampleid

        elif is_training == 'inference':
            labs = self.labels_multihot[self.trainidx[self.currenttestidx]]
            feat = self.features[self.trainidx[self.currenttestidx]]
            vn = self.videonames[self.trainidx[self.currenttestidx]]
            if self.currenttestidx == len(self.trainidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            return feat, np.array(labs),vn, done

        else:
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            feat = self.features[self.testidx[self.currenttestidx]]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            clip_feat = np.load(os.path.join(self.clip_path, 'test', self.videonames[self.testidx[self.currenttestidx]].decode('utf-8')+'.npy'))
            if self.currenttestidx == len(self.testidx) - 1:
                done = True
                self.currenttestidx = 0
            else:
                done = False
                self.currenttestidx += 1
            feat = np.array(feat)
            clip_feat = np.array(clip_feat)
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
        
            return feat, clip_feat, self.temp_label[self.testidx[self.currenttestidx]], np.array(labs), vn, done
        
    def random_avg(self, x, segm=None):
        if len(x) < self.num_segments:
            ind = self.random_perturb(len(x))
            x_n = x[ind]
            segm = segm[ind] if segm is not None else None
            return x_n, segm
        else:
            inds = np.array_split(np.arange(len(x)), self.num_segments)
            x_n = np.zeros((self.num_segments, x.shape[-1])).astype(x.dtype)
            segm_n = np.zeros(
                (self.num_segments, segm.shape[-1])).astype(x.dtype)
            for i, ind in enumerate(inds):
                x_n[i] = np.mean(x[ind], axis=0)
                if segm is not None:
                    segm_n[i] = segm[(ind[0] + ind[-1]) // 2]
            return x_n, segm_n if segm is not None else None

    def random_pad(self, x, segm=None):
        length = self.num_segments
        if x.shape[0] > length:
            strt = np.random.randint(0, x.shape[0] - length)
            x_ret = x[strt:strt + length]
            if segm is not None:
                segm = segm[strt:strt + length]
                return x_ret, segm
        elif x.shape[0] == length:
            return x, segm
        else:
            pad_len = length - x.shape[0]
            x_ret = np.pad(x, ((0, pad_len), (0, 0)), mode='constant')
            if segm is not None:
                segm = np.pad(segm, ((0, pad_len), (0, 0)), mode='constant')
            return x_ret, segm

    def random_perturb(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(
                        range(int(samples[i]),
                              int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(
                        range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)

    def uniform_sampling(self, length):
        if self.num_segments == length:
            return np.arange(self.num_segments).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)