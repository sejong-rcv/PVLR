from __future__ import print_function
import numpy as np
import utils.wsad_utils as utils
import random
import os
import options

class AntSampleDataset:
    """
    Dataset class for ActivityNet 1.3
    """
    def __init__(self, args, mode="both",sampling='random'):
        self.dataset_name = args.dataset_name

        self.num_class = args.num_class  # 200
        self.sampling = sampling
        self.num_segments = args.max_seqlen
        self.feature_size = args.feature_size  # 2048
        self.path_to_features = args.path_dataset
        self.path_to_annotations = os.path.join(args.path_dataset, self.dataset_name + "-Annotations/")

        self.segments = np.load(
            self.path_to_annotations + "segments.npy", allow_pickle=True
        )
        self.labels = np.load(
            self.path_to_annotations + "labels_all.npy", allow_pickle=True
        )
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
        self.clip_path = args.path_clip_dataset
        self.batch_size = args.batch_size
        self.t_max = args.max_seqlen
        
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

    def train_test_idx(self):
        for i, s in enumerate(self.subset):
            if s == "train":
                self.trainidx.append(i)
            elif s == "val":
                self.testidx.append(i)

    def classwise_feature_mapping(self):
        for category in self.classlist:
            idx = []
            for i in self.trainidx:
                for label in self.labels[i]:
                    if label == category:
                        idx.append(i)
                        break
            self.classwiseidx.append(idx)

    def load_data(self, n_similar=0, is_training=True, similar_size=2):
        if is_training:
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
            clip_feat = []

            for i in idx:
                try:
                    ifeat = np.load(os.path.join(self.path_to_features, "train", self.videonames[i]+'.npy'))
                except:
                    ifeat = np.load(os.path.join(self.path_to_features, "test", self.videonames[i]+'.npy'))
                clip_ifeat = np.load(os.path.join(self.clip_path, self.videonames[i]+'.npy'))
                if self.sampling == 'random':
                    sample_idx = self.random_perturb(ifeat.shape[0])
                elif self.sampling == 'uniform':
                    sample_idx = self.uniform_sampling(ifeat.shape[0])
                elif self.sampling == "all":
                    sample_idx = np.arange(ifeat.shape[0])
                else:
                    raise AssertionError('Not supported sampling !')
                if not ifeat.shape[0] == clip_ifeat.shape[0]:
                    import pdb;pdb.set_trace()

                ifeat = ifeat[sample_idx]
                clip_ifeat = clip_ifeat[sample_idx]

                feat.append(ifeat)
                clip_feat.append(clip_ifeat)
            feat = np.array(feat)
            clip_feat = np.array(clip_feat)
            labels = np.array([self.labels_multihot[i] for i in idx])
            if self.mode == "rgb":
                feat = feat[..., : self.feature_size]
            elif self.mode == "flow":
                feat = feat[..., self.feature_size :]
            return feat, clip_feat, labels, rand_sampleid

        else: # test
            labs = self.labels_multihot[self.testidx[self.currenttestidx]]
            vn = self.videonames[self.testidx[self.currenttestidx]]
            try:
                feat = np.load(os.path.join(self.path_to_features, "train", vn+'.npy'))
            except:
                feat = np.load(os.path.join(self.path_to_features, "test", vn+'.npy'))
            clip_feat = np.load(os.path.join(self.clip_path, self.videonames[self.testidx[self.currenttestidx]]+'.npy'))
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
            return feat, clip_feat, np.array(labs), vn, done
        

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