import torch
import os
from PIL import Image
from PIL import ImageFile
from torchvision import transforms

from torch.utils.data import Dataset
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True 
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImageDataset(Dataset):
    def __init__(self, 
                 dataset_path='/path/to/mvtec',
                 class_name='bottle', 
                 is_train=True,
                 resize=256,
                 cropsize=224,
                 synthetic=False,
                 feat_select=False):
        
        super().__init__()

        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.resize = (resize, resize)
        self.cropsize = cropsize
        self.synthetic = synthetic

        if self.is_train:
            if self.synthetic:
                self.x, self.mask = self.load_dataset_folder()
            else:
                self.x = self.load_dataset_folder()
        else:
            self.x, self.y, self.mask = self.load_dataset_folder()

        self.transform_img = transforms.Compose([transforms.Resize(self.resize),
                                                 transforms.CenterCrop(self.cropsize),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=IMAGENET_MEAN,
                                                                      std=IMAGENET_STD)])
        self.transform_mask = transforms.Compose([transforms.Resize(self.resize, Image.NEAREST),
                                                  transforms.CenterCrop(self.cropsize),
                                                  transforms.ToTensor()])

        self.transform_patch = transforms.Compose([transforms.Resize(self.resize, Image.NEAREST),
                                                  transforms.CenterCrop(self.cropsize),
                                                  transforms.Resize((int(self.cropsize / 8), int(self.cropsize / 8))),
                                                  transforms.ToTensor()])

        self.feat_select = feat_select
                                                  
    def __len__(self):
        return len(self.x)
  
    def __getitem__(self, idx):
        x = self.x[idx]
        x = Image.open(x).convert('RGB') # PIL.Image.Image

        x = self.transform_img(x)

        if self.is_train:
            if self.synthetic:
                mask = self.mask[idx]
                mask = Image.open(mask).convert('L')

                if self.feat_select == False:
                    mask = self.transform_patch(mask)
                    mask = torch.ceil(mask)
                    mask = mask.type(torch.int32)
                else:
                    mask = self.transform_mask(mask)

                return x, mask
            else:
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
        if not self.synthetic:
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
                img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if (f.endswith('.png') | f.endswith('.JPG') | f.endswith('.jpg'))])
                
                if self.is_train:
                    x.extend(img_fpath_list)
                else:
                    if img_type == 'good':
                        x.extend(img_fpath_list)
                        y.extend([0] * len(img_fpath_list))
                        mask.extend([None] * len(img_fpath_list))
        
                    else:
                        gt_type_dir = os.path.join(gt_dir, img_type)
                        img_fname_list = os.listdir(gt_type_dir)
                        gt_fpath_list = sorted([os.path.join(gt_type_dir, img_fname) for img_fname in img_fname_list])

                        x.extend(img_fpath_list)
                        y.extend([1] * len(img_fpath_list))
                        mask.extend(gt_fpath_list)
        else:
            img_type_dir = os.path.join(self.dataset_path, 'image', self.class_name)
            mask_type_dir = os.path.join(self.dataset_path, 'mask', self.class_name)
            if not os.path.isdir(img_type_dir):
                pass
            x = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if (f.endswith('.png') | f.endswith('.JPG') | f.endswith('.jpg'))])
            mask = sorted([os.path.join(mask_type_dir, f) for f in os.listdir(mask_type_dir) if (f.endswith('.png') | f.endswith('.JPG') | f.endswith('.jpg'))])

        if self.is_train:
            if self.synthetic:
                return x, mask
            else:
                return list(x)
        else:
            return list(x), list(y), list(mask)