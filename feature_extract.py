import dataset_extract
import torch
import argparse
from torch.utils.data import DataLoader
import numpy as np
import os

def set_device():
    use_cuda = torch.cuda.is_available()
    return torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('Feature_extract')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--feature_path', type=str)
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'], default='mvtec')
    parser.add_argument('--noise', type=str, choices=['0%', '1%', '2%', '3%', '5%', '10%', '20%'], default='10%')   
    parser.add_argument('-d', '--subdataset', type=str, default='pill')
    return parser.parse_args()

def extract_feature(input, feature_extractor):
    with torch.no_grad():
        feature_extractor.eval()
        feature = feature_extractor.get_intermediate_layers(input)[0]

    x_norm = feature[:, 0, :]
    x_norm = x_norm.squeeze()

    x_prenorm = feature[:, 1:, :]
    x_prenorm = x_prenorm.squeeze()

    if x_prenorm.shape[1] != 784:
        x_prenorm = x_prenorm.unsqueeze(0)
        x_norm = x_norm.unsqueeze(0)

    x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
    x_prenorm = torch.cat([x_norm, x_prenorm], axis=-1)
    x_prenorm = x_prenorm.squeeze()

    return x_prenorm

def extract_and_save_features(feature_extractor, loader, save_path, class_name, noise, is_train=True):
    os.makedirs(save_path, exist_ok=True)
    device = set_device()

    if is_train:
        features = []
        for x in loader:
            x = x.to(device)
            feature = extract_feature(x, feature_extractor)
            feature = feature.detach().cpu().numpy()

            features.append(feature)
        
        features = np.stack(features)
        if noise != '10%':
            os.makedirs(os.path.join(save_path, noise), exist_ok=True)
            np.save(os.path.join(save_path, noise, class_name+'.npy'), features)
        else:
            np.save(os.path.join(save_path, class_name+'.npy'), features)

    else:
        features = []
        label = []
        gt_mask = []

        for x, y, mask in loader:
            x = x.to(device)
            feature = extract_feature(x, feature_extractor)
            feature = feature.detach().cpu().numpy()

            features.append(feature)
            label.append(y.detach().numpy())
            gt_mask.append(mask.squeeze().detach().numpy())

        features = np.stack(features)
        gt = np.concatenate(label)
        mask = np.stack(gt_mask)

        np.save(os.path.join(save_path, class_name+'_test.npy'), features)
        np.save(os.path.join(save_path, class_name+'_gt.npy'), gt)
        np.save(os.path.join(save_path, class_name+'_mask.npy'), mask)

def main():
    args = parse_args()
    device = set_device()

    feature_extractor = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    feature_extractor = feature_extractor.to(device)

    train_set = dataset_extract.MyDataset(dataset_path=args.data_path, dataset=args.dataset, class_name=args.subdataset, is_train=True)
    train_loader = DataLoader(train_set, batch_size=1, pin_memory=True)

    extract_and_save_features(feature_extractor, train_loader, args.feature_path, args.subdataset, args.noise, is_train=True)

    if (args.dataset == 'mvtec') and (args.noise == '10%'):
        test_set = dataset_extract.MyDataset(dataset_path=args.data_path, dataset=args.dataset, class_name=args.subdataset, is_train=False)
        test_loader = DataLoader(test_set, batch_size=1, pin_memory=True)

        extract_and_save_features(feature_extractor, test_loader, args.feature_path, args.subdataset, args.noise, is_train=False)

if __name__ == "__main__":
    main()
