"""This is an adaptation of code TinyImageNet loading/processing code from 

https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54
"""

import os
import imageio
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while(img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir):
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path, wnids_path, words_path)
  
    def _make_paths(self, 
                    train_path, 
                    val_path, 
                    test_path, 
                    wnids_path, 
                    words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],    # [img_path, id, nid, box]
            'val':   [],    # [img_path, id, nid, box]
            'test':  []     # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x), os.listdir(test_path)))
        
        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 mode='train', 
                 preload=True, 
                 transform=None, 
                 download=False, 
                 max_samples=None):
        tinp = TinyImageNetPaths(root_dir)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []
        self.classes = list(np.arange(0,200))

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE, dtype=np.uint8)
      
            self.label_data = np.zeros((self.samples_num,), dtype=int)
            for idx in range(self.samples_num):
                s = self.samples[idx]
                img = np.array(imageio.imread(s[0]))
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = np.array(imageio.imread(s[0]))
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]
        
        if self.transform:
            img = self.transform(img)
        return img, lbl