import torch
import argparse
import numpy as np
import random 
import torch.backends.cudnn as cudnn
import faiss
import os
import wandb
import sys
import model
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import cv2
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve
import pandas as pd 
import datetime
import inference 
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pdb
import dataload
import math
#from info_nce import InfoNCE, info_nce
from skimage import morphology
import imgaug.augmenters as iaa
from torchvision import transforms
from PIL import Image
import timm
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

import sys
print(sys.executable)

matplotlib.use('Agg')
# device setup
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def parse_args():
    parser = argparse.ArgumentParser('self-train_ad')
    parser.add_argument('--data_path', type=str, default='/path/to/mvtec')
    parser.add_argument('--save_path', type=str, default='/path/to/save_path')
    parser.add_argument('--feature_path', type=str, default='/path/to/DINO')
    parser.add_argument('--synthetic_path', type=str, default='/path/to/sdas_perlin_10')
    parser.add_argument('--kl', action='store_true')
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--beta', action='store_true')
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--hist', action='store_true')
    parser.add_argument('--dataset', type=str, choices=['mvtec', 'visa'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-l', '--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-r', '--random', type=float, default=0.5)
    parser.add_argument('-t', '--threshold', type=float, default=0.5)
    parser.add_argument('-n', '--noise_threshold', type=float, default=0.9)
    parser.add_argument('-d', '--subdataset', type=str, default=None)
    parser.add_argument('--noise', type=str, default='10%', choices=['0%', '1%','2%', '3%','5%', '10%', '20%'])
    parser.add_argument('--std', type=float, default=None)
    parser.add_argument('--k_number', type=int, default=2)
    parser.add_argument('--llambda', type=float, default=1)
    parser.add_argument('--weight', type=float, default=0)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--beta_number', type=int, default=15)
    parser.add_argument('--alternative', action='store_true')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--balancing', action='store_true')
    parser.add_argument('--ratio', type=float, default=0.1)
    parser.add_argument('--oto_loss', type=str, choices=['kl', 'mae', 'mse'], default='mae')
    parser.add_argument('--perlin', action='store_true')
    parser.add_argument('--perlin_ratio', type=str, default='10%', choices=['1%', '5%', '10%', '25%', '50%', '100%'])
    parser.add_argument('--eval_interval', type=float, default=1)
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--backbone_type', type=str, default='vit')

    return parser.parse_args()

def find_matching(id):
    matching = [[], []]
    for i in range(id.shape[0]):
        if i in matching[1]:
            continue
        if i == id[id[i]]:
            input = i
            target = id[i]

            matching[0].append(input)
            matching[1].append(target)

    return matching

def compute_distance(feature):
    faiss.omp_set_num_threads(4)
    index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), 
                                 feature.shape[-1], 
                                 faiss.GpuIndexFlatConfig())
    index.add(feature)

    embedding = np.ascontiguousarray(feature)
    distance, id = index.search(embedding, k=2)

    distance = distance.T
    distance = distance[-1]

    id = id.T[-1]
    distance = np.expand_dims(distance, axis=-1)
    return distance, id

def fix_seed(number):
    np.random.seed(number)
    random.seed(number)
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.cuda.manual_seed_all(number)
    cudnn.benchmark = False
    cudnn.deterministic = True

def extract_feature(input, feature_extractor, concat):
    with torch.no_grad():
        feature_extractor.eval()
        feature = feature_extractor.get_intermediate_layers(input)[0]

    x_prenorm = feature[:, 1:, :] # patch tokens
    x_prenorm = x_prenorm.squeeze()

    if concat:
        x_norm = feature[:, 0, :] # [cls] tokens
        x_norm = x_norm.squeeze()

    if x_prenorm.shape[1] != 784:
        x_prenorm = x_prenorm.unsqueeze(0)
        x_norm = x_norm.unsqueeze(0)

    if concat:
        x_norm = torch.repeat_interleave(x_norm.unsqueeze(1), x_prenorm.shape[1], dim=1)
        x_prenorm = torch.cat([x_norm, x_prenorm], axis=-1)

    return x_prenorm

def save_model(model, saved_dir, weight_name):
    os.makedirs(saved_dir, exist_ok=True)
    check_point = {
        'net': model.state_dict()
    }
    torch.save(check_point, os.path.join(saved_dir, weight_name))

def set_wandb(args):
    wandb.init(project=args.dataset+"_"+args.subdataset,
               config={"learning_rate": args.lr,
                       "num_epochs": args.epoch,
                       "batch_size": args.batch_size,
                       "threshold": args.threshold,
                       "random": args.random,
                       "noise": args.noise,
                       "overlap": args.overlap,
                       "balancing": args.balancing,
                       "oto": args.kl,
                       "oto_loss": args.oto_loss,
                       "gaussian": args.gaussian,
                       "weight": args.weight,
                       "synthetic": args.synthetic,
                       },
                name="gaussian_"+str(args.gaussian)+"_noise_"+args.noise+"_balancing_"+str(args.balancing)+"_oto_"+str(args.kl)+"_weight_"+str(args.weight)+"_synthetic_"+str(args.synthetic)
                )


def main():
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    tsne = TSNE(n_components=2, random_state=args.seed)
    if args.synthetic or args.perlin:
        if args.backbone_type == 'vit':
            backbone = torch.hub.load("facebookresearch/dino:main", "dino_vitb8").to(device)
        elif args.backbone_type == 'wideresnet':
            outlayers = ['layer1', 'layer2', 'layer3', 'layer4']
            layers_idx ={'layer1':1, 'layer2':2, 'layer3':3, 'layer4':4}
            backbone = timm.create_model('wide_resnet50_2', features_only=True,pretrained=True,
                                            out_indices=[layers_idx[outlayer] for outlayer in outlayers])
            resnet_channel = 1024
            resnet_idx = 2
    # torch.cuda.set_device(args.gpu)
    
    CLASS_NAMES = []
    if args.subdataset == None:
        if args.dataset == 'mvtec':
            CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
        
        else:
            CLASS_NAMES = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1',
                        'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    
    else:
        CLASS_NAMES.append(args.subdataset)


    localnet = model.localnet(len_feature=1536)
    localnet = localnet.to(device)
    
    localnet_optimizer = optim.RMSprop(localnet.parameters(), lr=args.lr, momentum=0.2)
    if args.alternative:
        onetoone_optimizer = optim.Adam(localnet.parameters(), lr=args.lr)
    localnet_criterion = nn.BCELoss().to(device)

    results = []

    for class_name in CLASS_NAMES:
        fix_seed(args.seed)
        if args.wandb:
            set_wandb(args)

        saved_dir = os.path.join(args.save_path, 'results', class_name)
        os.makedirs(saved_dir, exist_ok=True)

        if args.noise != '10%':
            train_features = np.load(os.path.join(args.feature_path, args.noise, class_name+'.npy'))
        else:
            if args.patch:
                train_features = np.load(os.path.join(args.feature_path, class_name+'_train_patch.npy'))
            else:
                train_features = np.load(os.path.join(args.feature_path, class_name+'.npy'))
        
        train_features = train_features.reshape(-1, 784, train_features.shape[-1])
        train_dataset = []


        # Store normal
        for i in range(train_features.shape[0]):
            train_dataset.append(train_features[i])


        if args.perlin:
            synthetic_dataset = dataload.ImageDataset(dataset_path=args.synthetic_path, class_name=class_name, synthetic=True)
            if args.perlin_ratio == '10%':
                perlin_length = int(len(train_dataset) * 0.1)
            elif args.perlin_ratio=='25%':
                perlin_length = int(len(train_dataset) * 0.25)
            elif args.perlin_ratio=='5%':
                perlin_length = int(len(train_dataset) * 0.05)
            elif args.perlin_ratio=='1%':
                perlin_length = int(len(train_dataset) * 0.01)
            elif args.perlin_ratio=='3%':
                perlin_length = int(len(train_dataset) * 0.03)
            
            synthetic_features = []
            synthetic_feat_torch = []
            for i in range(perlin_length):
                image, mask = synthetic_dataset[i] # [3, 224, 224]
                image = image.unsqueeze(0).cuda()
                if args.backbone_type == 'vit':
                    _syn_feature = extract_feature(image, backbone, True) # [B, 784, 1536]
                else:
                    image = image.cpu()
                    with torch.no_grad():
                        _syn_feature = backbone(image)[resnet_idx].reshape(1, resnet_channel, -1).permute(0,2,1)

                        # chgd
                        _syn_feature = F.interpolate(_syn_feature.reshape(1, 14, 14, 1024).permute(0,3,1,2), \
                            size=(28,28), mode='bilinear', align_corners=False).reshape(1, 1024, -1).permute(0,2,1)

                dim = _syn_feature.shape[-1]
                _syn_feature = _syn_feature.cpu()
                _syn_feature = _syn_feature.reshape(-1, dim)

                synthetic_feat_torch.append(_syn_feature)
                synthetic_features.append(_syn_feature.cpu().detach().numpy())

            final_synthetic_feat = torch.stack(synthetic_feat_torch)
            initial_train_len = len(train_dataset)
            syn_idx = 0

            # Store anomaly
            with torch.no_grad():
                cri_sum_vec_syn = list()
                for i in range(perlin_length):
                    train_dataset.append(synthetic_features[i])
                    syn_idx += 1
        
        if args.synthetic:
            synthetic_dataset = dataload.ImageDataset(dataset_path=args.synthetic_path, class_name=class_name, synthetic=True)

        if args.perlin:
            local_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, drop_last=True, num_workers=16)
        else:
            local_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, drop_last=True, num_workers=16)
        
        mini_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
        
        if args.overlap:
            test_features = np.load(os.path.join(args.feature_path, class_name+'_test.npy')).squeeze()
            mask_gt = np.load(os.path.join(args.feature_path, class_name+'_mask.npy')).squeeze()
            mask_gt = np.ceil(mask_gt)
            label_gt = np.load(os.path.join(args.feature_path, class_name+'_gt.npy'))
        else:
            if args.patch:
                test_features = np.load(os.path.join(args.feature_path, class_name+'_test_patch.npy')).squeeze()
                mask_gt = np.load(os.path.join(args.feature_path, class_name+'_test_patch_masks.npy')).squeeze()
                mask_gt = np.ceil(mask_gt)
                label_gt = np.zeros(mask_gt.shape[0])
                label_gt[np.unique(np.where(mask_gt == 1)[0])] = 1
            else:
                test_features = np.load(os.path.join(args.feature_path, class_name+'_test.npy')).squeeze()
                mask_gt = np.load(os.path.join(args.feature_path, class_name+'_mask.npy')).squeeze()
                mask_gt = np.ceil(mask_gt)
                label_gt = np.load(os.path.join(args.feature_path, class_name+'_gt.npy'))


        
        test_dataset = []
        for i in range(test_features.shape[0]):
            test_dataset.append([test_features[i], label_gt[i], mask_gt[i]])
        test_loader = DataLoader(test_dataset, batch_size=512, pin_memory=True)


        if args.oto_loss == 'mae':
            l_loss = nn.L1Loss().to(device)
        elif args.oto_loss == 'mse':
            l_loss = nn.MSELoss().to(device)
        else:
            l_loss =  nn.KLDivLoss(reduction='batchmean').to(device)
        
        total_batch = len(local_loader)
        threshold = args.threshold
        noise_threshold = args.noise_threshold

        iteration = 0
        if args.beta:
            beta = torch.distributions.beta.Beta(0.5*torch.ones(784), 0.5*torch.ones(784))
            
        for epoch in range(args.epoch):
            local_loss = 0
            oto_loss = 0
            bce_loss = 0

            if args.perlin:
                original_indices = list(range(len(train_dataset)))
                shuffled_indices = np.random.permutation(original_indices)
                index_mapping = dict(zip(original_indices, shuffled_indices))
                perlin_indices = original_indices[-perlin_length:]

            data_counter = 0
            for x in tqdm.tqdm(local_loader,  '| run | train | '+str(epoch+1)+' |'):
                batch = x.shape[0]
                if args.perlin:
                    shuffled_dataloader = []
                    original_indices_batch = range(data_counter * local_loader.batch_size, (data_counter + 1) * local_loader.batch_size)
                    shuffled_indices_batch = [index_mapping[original_index] for original_index in original_indices_batch]
                    shuffled_batch = [torch.tensor(train_dataset[shuffled_index]) for shuffled_index in shuffled_indices_batch]
                    shuffled_dataloader.extend(shuffled_batch)
                    x = torch.stack(shuffled_dataloader)
                    data_counter += 1

                if threshold <= 1:
                    with torch.no_grad():
                        _f_stack = []
                        _s_stack = []
                        if args.beta:
                            _x_stack = []

                        for mini_x in mini_loader:
                            features, score = localnet(mini_x.to(device))
                            if features.shape[0] > args.batch_size:
                                features = features.unsqueeze(0)

                            features = features.detach().cpu().numpy()
                            score = score.detach().cpu().numpy()
                            score = score.max(axis=-1)

                            if score.shape == ():
                                score = np.array([score], dtype=np.float32)

                            _f_stack.append(features)
                            _s_stack.append(score)
                            if args.beta:
                                _x_stack.append(mini_x)

                        _f_stack = np.concatenate(_f_stack)
                        _s_stack = np.concatenate(_s_stack)
                    
                        _s_stack = (_s_stack - _s_stack.min())/(_s_stack.max() - _s_stack.min())
                        _idx = np.where(_s_stack < 0.5)[0]
                        
                        if args.random < 1:
                            num = int(_idx.shape[0] * args.random)
                            idx = []
                            for i in range(num):
                                a = np.random.randint(_idx.shape[0])
                                while _idx[a] in idx:
                                    a = np.random.randint(_idx.shape[0])
                                idx.append(_idx[a])                       

                        else:
                            idx = _idx         

                        normal_features = _f_stack[idx]
                        normal_features = normal_features.reshape(-1, normal_features.shape[-1])
                        dim = int(normal_features.shape[-1])

                        faiss.omp_set_num_threads(4)
                        index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(),
                                                    dim,
                                                    faiss.GpuIndexFlatConfig())
                        index.add(normal_features)

                        if args.beta:
                            first_vector = []
                            second_vector = []

                            _x_stack = torch.concat(_x_stack)
                            _x_stack = _x_stack[_s_stack > 0.5]
                            _x_stack = _x_stack.reshape(-1, dim)

                            high_score_features = _f_stack[_s_stack > 0.5]
                            high_score_features = high_score_features.reshape(-1, dim)

                            high_score_distance, _ = index.search(np.ascontiguousarray(high_score_features), k=1)
                            high_score_distance = high_score_distance.T
                            high_score_distance = high_score_distance.squeeze()
                            
                            confident_anomaly = np.argsort(high_score_distance)[::-1][:args.beta_number]
                            confident_anomaly = confident_anomaly.tolist()
                            confident_features = _x_stack[confident_anomaly]

                            for i in range(784):
                                first = random.randint(0, args.beta_number-1)
                                second = random.randint(0, args.beta_number-1)
                                while second == first:
                                    second = random.randint(0, args.beta_number-1)
                                first_vector.append(confident_features[first])
                                second_vector.append(confident_features[second])

                            first_vector = torch.stack(first_vector)
                            second_vector = torch.stack(second_vector)

                            syn_anomaly = a * first_vector + (1-a) * second_vector
                            syn_anomaly = syn_anomaly.unsqueeze(0)
                        
                        features, _ = localnet(x.to(device))
                        features = features.detach().cpu().numpy()
                        features = features.reshape(-1, features.shape[-1])

                        if (iteration >= args.iter) and args.kl:
                                
                            _, id = compute_distance(features)
                            matched_id = find_matching(id)

                        anomaly_embedding = np.ascontiguousarray(features)
                        distance, _ = index.search(anomaly_embedding, k=args.k_number)

                        distance[distance < 1e-2] = 0
                        distance = distance.T
                        distance = distance.squeeze()

                        if args.k_number == 2:
                            same_feature = distance[0] == 0
                            distance[0][same_feature] = distance[1][same_feature]
                            distance = distance[0]

                        if distance.max() == distance.min():
                            distance = np.zeros(distance.shape)
                        else:
                            maximum = distance.max()
                            minimum = distance.min()
                            distance = (distance - minimum) / (maximum - minimum)

                        distance = distance.reshape(-1, 784)

                        if (args.threshold <= 1) and args.gaussian:
                            _idx = (distance > threshold) & (distance < noise_threshold)
                            _idx = _idx.reshape(-1)
                            n = np.where(_idx)[0].shape[0]
                            _copy = x.detach()
                            
                            if n > 0:
                                if args.std == None:
                                    std = _copy.std(axis=0).max(axis=0)[0]
                                    std = std.repeat_interleave(n).reshape(-1, n)
                                    std = std.T
                                    _copy = _copy.reshape(-1, dim)
                                    _copy[_idx] += torch.normal(mean=0, std=std)
                                else:
                                    _copy = _copy.reshape(-1, dim)
                                    _copy[_idx] += torch.normal(mean=0, std=std, size=_copy[_idx].shape)

                            _copy = _copy.reshape(-1, 784, dim)

                else:
                    dim = x.shape[-1]
                if args.beta:
                    local_label = torch.zeros((args.batch_size + 1, 784))
                    distance[distance > threshold] = 1
                    distance[distance <= threshold] = 0
                    local_label[:-1] = torch.tensor(distance)
                    if (args.threshold <= 1) and args.gaussian:
                        _copy = torch.concat([_copy, syn_anomaly])
                    else:
                        x = torch.concat([x, syn_anomaly])
                
                else:
                    local_label = torch.zeros((args.batch_size, 784))
                    if args.threshold <= 1:
                        local_label[distance > threshold] = 1
                        
                    if args.synthetic:
                        synthetic_ids = []
                        synthetic_features = []

                        perlin_length = int(len(train_dataset) * 0.1)
                        synthetic_loop = int(perlin_length / (len(train_dataset) / args.batch_size))

                        for i in range(synthetic_loop):
                            a = np.random.randint(len(synthetic_dataset))
                            while a in synthetic_ids:
                                a = np.random.randint(len(synthetic_dataset))
                            synthetic_ids.append(a)
                            image, mask = synthetic_dataset[a]
                            image = image.unsqueeze(0).to(device)
                            _syn_feature = extract_feature(image, backbone, True)
                            _syn_feature = _syn_feature.cpu()
                            mask = mask.reshape(-1)
                            _syn_feature = _syn_feature.reshape(-1, dim)
                            synthetic_features.append(_syn_feature[mask == 1])
                        
                        synthetic_features = torch.cat(synthetic_features, dim=0)
                        local_label = local_label.reshape(-1)
                        local_label = torch.cat([local_label, torch.ones(synthetic_features.shape[0])])

                        x = x.reshape(-1, dim)
                        x = torch.cat([x, synthetic_features])

                        if (args.threshold <= 1) and args.gaussian:
                            _copy = _copy.reshape(-1, dim)
                            _copy = torch.cat([_copy, synthetic_features])

                    if (args.threshold <= 1) and args.hist:
                        fig, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
                        axes.hist(distance.reshape(-1), density=True, bins=100, alpha=1)

                        axes.set_title(class_name)
                        axes.set_xlabel("local feature distance")
                        axes.set_ylabel("density")

                        fig.savefig(os.path.join('plot', class_name, 'distance_'+str(iteration)+'.png'), dpi=300, format='png', bbox_inches='tight')
                        plt.close()

                localnet.train()
                localnet_optimizer.zero_grad()
                if args.alternative:
                    onetoone_optimizer.zero_grad()

                x = x.to(device)
                local_label = local_label.to(device)

                _, local_pred = localnet(x)


                if (args.threshold <= 1) and args.gaussian:
                    _copy = _copy.to(device)
                    _, gaussian_pred = localnet(_copy)
                    if args.balancing:
                        if np.where(distance)[0].shape[0] == 0:
                            _a_loss = 0
                        else:
                            _a_loss = localnet_criterion(gaussian_pred[local_label == 1], local_label[local_label == 1])
                        _n_loss = localnet_criterion(gaussian_pred[local_label == 0], local_label[local_label == 0])
                        _loss = _a_loss + _n_loss
                    else:
                        _loss = localnet_criterion(gaussian_pred, local_label)
                else:
                    if args.balancing:
                        if np.where(distance)[0].shape[0] == 0:
                            _a_loss = 0
                        else:
                            _a_loss = localnet_criterion(local_pred[local_label == 1], local_label[local_label == 1])
                        _n_loss = localnet_criterion(local_pred[local_label == 0], local_label[local_label == 0])
                        _loss = _a_loss + _n_loss
                    else:
                        _loss = localnet_criterion(local_pred, local_label)
                # _loss = localnet_criterion(local_pred, local_label)

                if (iteration >= args.iter) and args.kl:
                    if args.synthetic:
                        real = args.batch_size * 784
                    else:
                        real = args.batch_size
                    target = local_pred[:real].reshape(-1)[matched_id[0]]
                    input = local_pred[:real].reshape(-1)[matched_id[1]]

                    if args.perlin:
                        for idx, shuf_idx in enumerate(shuffled_indices_batch):
                            if shuf_idx in perlin_indices:
                                target[idx] = target[idx].detach()
                                input[idx] = input[idx].detach()

                    if args.oto_loss == 'kl':
                        _l_loss = 0.5 * (l_loss(input.log(), (input + target)/2) + l_loss(target.log(), (input + target)/2))
                    else:
                        _l_loss = 0.5 * (l_loss(input, (input + target)/2) + l_loss(target, (input + target)/2))
                else:
                    _l_loss = 0
                
                if args.alternative:
                    _local_loss = _loss

                else:
                    _local_loss = _loss + args.weight * _l_loss

                try:
                    _local_loss.backward()
                    localnet_optimizer.step()

                    if args.alternative:
                        _l_loss.backward()
                        onetoone_optimizer.step()
                        
                except Exception as err:
                    pdb.set_trace()

                local_loss += _local_loss/total_batch
                bce_loss += _loss/total_batch
                oto_loss += _l_loss/total_batch
                iteration += 1

            seg_map = []
            img_map = []
            label_gt = []
            mask_gt = []

            if (epoch) % args.eval_interval == 0:
                for x, y, mask in test_loader:
                    with torch.no_grad():
                        localnet.eval()
                        x = x.to(device)
                        y = y.detach().numpy()
                        mask = mask.detach().numpy()
                        _, score = localnet(x)
                        score = score.detach().cpu().numpy()

                        img_score = score.max(axis=1)
                        img_score = img_score
                        img_map.append(img_score)

                        score = score.reshape(-1, 28, 28)
                        for i in range(score.shape[0]):
                            _map = cv2.resize(score[i], (224, 224))
                            _map = gaussian_filter(_map, sigma=4)
                            seg_map.append(_map)

                            label_gt.append(y[i])
                            mask_gt.append(mask[i])
                    
                img_map = np.concatenate(img_map, axis=0)
                label_gt = np.array(label_gt)
                seg_map = np.stack(seg_map, axis=0)
                mask_gt = np.stack(mask_gt, axis=0)
                    
                auroc = roc_auc_score(label_gt, img_map)
                pixel_auroc = roc_auc_score(mask_gt.ravel(), seg_map.ravel())
                num_epoch = epoch + 1

                print('epoch %d |' % num_epoch, f'auroc: {auroc:.5f}, pxiel auroc: {pixel_auroc:.5f}')

                if args.wandb:
                    wandb.log({'total loss': local_loss,
                            'one-to-one loss': oto_loss,
                            'bce loss': bce_loss,
                            'image AUC': auroc,
                            'pixel AUC': pixel_auroc})

                mean = (auroc + pixel_auroc)/2
                
                if epoch == 0:
                    best = mean
                    fix_auroc = auroc
                    fix_pauroc = pixel_auroc
                else:
                    if mean > best:
                        best = mean
                        fix_auroc = auroc
                        fix_pauroc = pixel_auroc
                        print(f"class: {class_name}, data_noise: {args.noise}")
                        print("curr_best_auc: ", fix_auroc)
                        print("curr_best_pauc: ", fix_pauroc)
                        save_model(localnet, saved_dir, "gaussian_"+str(args.gaussian)+"_noise_"+args.noise+"_balancing_"+str(args.balancing)+"_oto_"+str(args.kl)+"_weight_"+str(args.weight)+"_synthetic_"+str(args.synthetic)+"_localnet.pt")
                if epoch % 100 == 0:
                    print("class: ", class_name)

                # Log
                if args.save_log:
                    directory = f'./log_5/{args.epoch}_epoch/{class_name}'
                    file_name = f'metrics.txt'
                    file_path = os.path.join(directory, file_name)
                    os.makedirs(directory, exist_ok=True)

                    with open(file_path, 'a') as file:
                        file.write(f'epoch {num_epoch} | auroc: {auroc:.5f}, pixel auroc: {pixel_auroc:.5f}\n')
                ##
                


    print("class:", class_name)
    print("fix_img_auroc:", fix_auroc)
    print("fix_pauroc:", fix_pauroc)

    results.append([class_name, fix_auroc, fix_pauroc])
    df = pd.DataFrame(results, columns=['class', 'auroc', 'pixel_auroc'])
    result_path = os.path.join(args.save_path, 'results', args.subdataset)
    os.makedirs(result_path, exist_ok=True)
    result_path = os.path.join(result_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(result_path, exist_ok=True)

    dir = os.path.join(result_path, "gaussian_"+str(args.gaussian)+"_noise_"+args.noise+"_balancing_"+str(args.balancing)+"_oto_"+str(args.kl)+"_weight_"+str(args.weight)+"_synthetic_"+str(args.synthetic)+"_result.xlsx")
    df.to_excel(dir, index=False) 

if __name__ == "__main__":
    main()