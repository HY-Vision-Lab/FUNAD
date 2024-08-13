from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import os
import random

class dataset(Dataset):
    def __init__(self, dataset_path='/home/jiin/Drive/dataset', data_name='MVTec_AD', score_path='/home/jiin/Drive/score',
                 class_name='bottle', is_train=True, pretrain=False, cropsize=224, resize=256, aug='mixup', threshold=0.5, noise_threshold=0.9):
        
        self.dataset_path = dataset_path
        self.class_name = class_name
        self.is_train = is_train
        self.data_name = data_name
        self.score_path = score_path
        self.resize = resize
        self.cropsize = cropsize
        self.pretrain = pretrain
        self.aug = aug
        self.threshold = threshold
        self.noise_threshold = noise_threshold

        self.y_exist = os.path.exists(os.path.join('/home/jiin/Desktop/glml/score', self.class_name+'_max_score.npy'))

        if self.is_train & (not self.y_exist or self.pretrain):
            self.x, self.mask = self.load_dataset_folder()
        elif self.is_train & self.y_exist:
            self.x, self.y, self.mask = self.load_dataset_folder()
        else:
            self.x, self.y, self.mask = self.load_dataset_folder()
        
        if self.is_train:
            if self.aug == 'mixup':
                self.ano = []
                for l in os.listdir('/home/jiin/Drive/dataset/dtd/images'):
                    for name in os.listdir(os.path.join('/home/jiin/Drive/dataset/dtd/images', l)):
                        if name.endswith('.jpg'):
                            self.ano.append(os.path.join('/home/jiin/Drive/dataset/dtd/images', l, name))
        
        self.transform_x = transforms.Compose([transforms.Resize(self.resize, Image.ANTIALIAS),
                                               transforms.CenterCrop(self.cropsize)])
        
        self.tensor_x = transforms.Compose([transforms.ToTensor(),
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

        mask = self.mask[idx]
        if self.is_train:
            mask = mask.reshape(28, 28)
            _idx = np.where((mask > self.threshold) & (mask < self.noise_threshold))
            synthetic = x

            if _idx[0].shape[0] != 0:
                h = 8 * _idx[0]
                w = 8 * _idx[1]
                if self.aug == 'cutpaste':
                    for i in range(h.shape[0]):
                        id = np.random.randint(len(self.x))
                        while id == idx:
                            id = np.random.randint(len(self.x))
                        synthetic = self.cutpaste(synthetic, id, h[i], w[i])
                
                else:
                    synthetic = self.mixup(synthetic, h, w)
            
            # synthetic.save(os.path.join('/Users/jiin/data', self.class_name, 'synthetic'+str(idx)+'.png'), "PNG")

            x = self.tensor_x(x)
            synthetic = self.tensor_x(synthetic)

            if (not self.pretrain) and self.y_exist:
                y = self.y[idx]
                return x, synthetic, y, mask
            else:
                return x, synthetic, mask

        else:
            x = self.tensor_x(x)
            y = self.y[idx]
            if y == 0:
                mask = torch.zeros([1, self.cropsize, self.cropsize])
            else:
                mask = Image.open(mask).convert('L')
                mask = self.transform_mask(mask)
                mask = torch.ceil(mask)

            return x, y, mask
        
    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        img_dir = os.path.join(self.dataset_path, self.data_name, self.class_name, phase)
        x = []

        if self.is_train & self.pretrain:
            mask = np.load(os.path.join(self.score_path, self.class_name+'_min_score.npy'))
            mask = (mask - mask.min())/(mask.max() - mask.min())

        elif self.is_train & (not self.pretrain):
            file = os.path.join('/home/jiin/Desktop/glml/score', self.class_name+'_iter_score.npy')
            y_file = os.path.join('/home/jiin/Desktop/glml/score', self.class_name+'_max_score.npy')
            if self.y_exist:
                mask = np.load(file)
                y = np.load(y_file)
            else:
                mask = np.load(os.path.join(self.score_path, self.class_name+'_min_score.npy'))
        
        else:
            x, y, mask = []
            gt_dir = os.path.join(self.dataset_path, self.data_name, self.class_name, 'ground_truth')
        
        img_types = sorted(os.listdir(img_dir))

        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue

            if self.data_name == 'MVTec_AD':
                img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.png')])
            else:
                img_fpath_list = sorted([os.path.join(img_type_dir, f) for f in os.listdir(img_type_dir) if f.endswith('.JPG')])
            
            x.extend(img_fpath_list)
            if not self.is_train:
                if img_type == 'good':
                    y.extend([0] * len(img_fpath_list))
                    mask.extend([None] * len(img_fpath_list))
                
                else:
                    gt_type_dir = os.path.join(gt_dir, img_type)
                    img_fname_list = os.listdir(gt_type_dir)
                    gt_fpath_list = sorted([os.path.join(gt_type_dir, img_fname) for img_fname in img_fname_list])

                    mask.extend(gt_fpath_list)

        if self.is_train & ((not self.y_exist) | self.pretrain):    
            return list(x), mask
        elif self.is_train & self.y_exist:
            return list(x), y, mask
        else:
            return list(x), list(y), list(mask)
        
    def cutpaste(self, x, idx, h, w):
        cut_img = self.x[idx]
        cut_img = Image.open(cut_img).convert('RGB')
        cut_img = self.transform_x(cut_img)
        
        size = (5, 5)

        from_location_h = int(random.uniform(0, h - size[0]))
        from_location_w = int(random.uniform(0, w - size[1]))

        box = [from_location_w, from_location_h, from_location_w + size[1], from_location_h + size[0]]
        patch = cut_img.crop(box)

        random_rotate = random.uniform(-45, 45)
        patch = patch.convert("RGBA").rotate(random_rotate, expand=True)
        mask = patch.split()[-1]

        aug_image = x.copy()
        aug_image.paste(patch, (w, h), mask=mask)

        return aug_image

    def mixup(self, x, h, w):
        size = (8, 8)
        aug_x = x

        for i in range(h.shape[0]):
            idx = np.random.randint(len(self.ano))
            cut_img = self.ano[idx]
            cut_img = Image.open(cut_img).convert('RGB')
            cut_img = self.transform_x(cut_img)

            from_location_h = int(random.uniform(0, h[i] - size[0]))
            from_location_w = int(random.uniform(0, w[i] - size[1]))

            box = [from_location_w, from_location_h, from_location_w + size[1], from_location_h + size[0]]
            patch = cut_img.crop(box)
            patch = patch.convert("RGBA")
            patch = self.adjust_transparency(patch, 0.4)

            mask = patch.split()[-1]
            aug_x = aug_x.copy()

            aug_x.paste(patch, (w[i], h[i]), mask=mask)
        
        return aug_x
    
    def adjust_transparency(self, image, transparency):
        alpha = image.split()[-1]
        alpha = alpha.point(lambda p: p * transparency)
        image.putalpha(alpha)

        return image