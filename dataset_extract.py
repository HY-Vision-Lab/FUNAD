import torch
import os
import random
from PIL import Image
from PIL import ImageFile
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True 

class MyDataset(Dataset):
    def __init__(self, dataset_path='data', dataset='mvtec',
                 class_name='pill', is_train=True, cropsize=224, resize=256):
        
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.cropsize = cropsize
        self.resize = (resize, resize)
        self.dataset = dataset

        if self.is_train:
            self.x = self.load_dataset_folder()
        else:
            self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_x = transforms.Compose([transforms.Resize(self.resize),
                                               transforms.CenterCrop(self.cropsize),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                    std=[0.229, 0.224, 0.225])])
      
        self.transform_mask = transforms.Compose([transforms.Resize(self.resize, Image.NEAREST),
                                                  transforms.CenterCrop(self.cropsize),
                                                  transforms.ToTensor()])

    def __len__(self):
        return len(self.x)
  
    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(x).convert('RGB')
        x = self.transform_x(x)

        if self.is_train:
            return x
        else:
            y = self.y[idx]
            mask = self.mask[idx]
            if y == 0:
                mask = torch.zeros([1, self.cropsize, self.cropsize], dtype=torch.int32)
            else:
                mask = Image.open(mask).convert('L')
                mask = self.transform_mask(mask)
                mask = torch.ceil(mask)
                mask = mask.type(torch.int32)

            return x, y, mask

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        img_dir = os.path.join(self.dataset_path, self.class_name, phase)

        if self.is_train:
            x, y = [], []
        else:
            gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')
            x, y, mask = [], [], []

        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            
            # if self.dataset == 'mvtec':
            #     end = '.png'
            # else:
            #     end = '.JPG'
            
            img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if (f.endswith('.png') | f.endswith('.JPG') | f.endswith('.jpg'))])

            if self.is_train:
                x.extend(img_fpath_list)

      # y labeling
            else:
                if img_type == 'good':
                    x.extend(img_fpath_list)
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
      
                else:
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    # img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                    img_fname_list = os.listdir(gt_type_dir)
                    gt_fpath_list = sorted([os.path.join(gt_type_dir, img_fname) for img_fname in img_fname_list])

                    x.extend(img_fpath_list)
                    y.extend([1] * len(img_fpath_list)) # abnormal
                    mask.extend(gt_fpath_list)

        if self.is_train:
            return list(x)
        else:
            return list(x), list(y), list(mask)

class Feature_set(Dataset):
    def __init__(self, feature_path, subclass, is_train):
        self.feature_path = feature_path
        self.subclass = subclass
        self.is_train = is_train
        if self.is_train:
            self.list = os.listdir(os.path.join(self.feature_path, self.subclass, 'train'))
        else:
            self.test_features = np.load(os.path.join(self.feature_path, self.subclass, 'test', 'test.npy'))
            self.test_gt = np.load(os.path.join(self.feature_path, self.subclass, 'test', 'gt.npy'))
            self.test_mask = np.load(os.path.join(self.feature_path, self.subclass, 'test', 'mask.npy'))
            self.test_mask = np.ceil(self.test_mask)

    def __len__(self):
        if self.is_train:
            return len(self.list)
        else:
            return len(self.test_features)

    def __getitem__(self, index):
        if self.is_train:
            dir = os.path.join(self.feature_path, self.subclass, 'train', self.list[index])
            features = np.load(dir, allow_pickle=True)

            return features
        else:
            return self.test_features[index], self.test_gt[index], self.test_mask[index]