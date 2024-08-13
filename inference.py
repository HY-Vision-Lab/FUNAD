import torch
import numpy as np
import sys
sys.path.append('/path/to/glml')
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
import os
import model
import tqdm
import cv2
from skimage import measure
from sklearn.metrics import auc
import matplotlib.pyplot as plt

def cal_pro_metric_new(labeled_imgs, score_imgs, fpr_thresh=0.3, max_steps=2000, class_name=None):
    labeled_imgs = np.array(labeled_imgs)
    labeled_imgs[labeled_imgs <= 0.45] = 0
    labeled_imgs[labeled_imgs > 0.45] = 1
    labeled_imgs = labeled_imgs.astype(np.bool_)
    score_imgs = np.array(score_imgs)

    max_th = score_imgs.max()
    min_th = score_imgs.min()
    delta = (max_th - min_th) / max_steps

    ious_mean = []
    ious_std = []
    pros_mean = []
    pros_std = []
    threds = []
    fprs = []
    binary_score_maps = np.zeros_like(score_imgs, dtype=np.bool_)
    for step in tqdm.tqdm(range(max_steps)):
        thred = max_th - step * delta
        # segmentation
        binary_score_maps[score_imgs <= thred] = 0
        binary_score_maps[score_imgs > thred] = 1

        pro = []  # per region overlap
        iou = []  # per image iou
        # pro: find each connected gt region, compute the overlapped pixels between the gt region and predicted region
        # iou: for each image, compute the ratio, i.e. intersection/union between the gt and predicted binary map
        for i in range(len(binary_score_maps)):  # for i th image
            # pro (per region level)
            label_map = measure.label(labeled_imgs[i], connectivity=2)
            props = measure.regionprops(label_map)
            for prop in props:
                x_min, y_min, x_max, y_max = prop.bbox
                cropped_pred_label = binary_score_maps[i][x_min:x_max, y_min:y_max]
                # cropped_mask = masks[i][x_min:x_max, y_min:y_max]
                cropped_mask = prop.filled_image  # corrected!
                intersection = np.logical_and(cropped_pred_label, cropped_mask).astype(np.float32).sum()
                pro.append(intersection / prop.area)
            # iou (per image level)
            intersection = np.logical_and(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            union = np.logical_or(binary_score_maps[i], labeled_imgs[i]).astype(np.float32).sum()
            if labeled_imgs[i].any() > 0:  # when the gt have no anomaly pixels, skip it
                iou.append(intersection / union)
        # against steps and average metrics on the testing data
        ious_mean.append(np.array(iou).mean())
        #             print("per image mean iou:", np.array(iou).mean())
        ious_std.append(np.array(iou).std())
        pros_mean.append(np.array(pro).mean())
        pros_std.append(np.array(pro).std())
        # fpr for pro-auc
        masks_neg = ~labeled_imgs
        fpr = np.logical_and(masks_neg, binary_score_maps).sum() / masks_neg.sum()
        fprs.append(fpr)
        threds.append(thred)

    # as array
    threds = np.array(threds)
    pros_mean = np.array(pros_mean)
    pros_std = np.array(pros_std)
    fprs = np.array(fprs)


    # default 30% fpr vs pro, pro_auc
    idx = fprs <= fpr_thresh  # find the indexs of fprs that is less than expect_fpr (default 0.3)
    fprs_selected = fprs[idx]
    fprs_selected = rescale(fprs_selected)  # rescale fpr [0,0.3] -> [0, 1]
    pros_mean_selected = pros_mean[idx]
    pro_auc_score = auc(fprs_selected, pros_mean_selected)
    # print("pro auc ({}% FPR):".format(int(expect_fpr * 100)), pro_auc_score)
    return pro_auc_score

def rescale(x):
    return (x - x.min()) / (x.max() - x.min())

device = 'cuda'

def main():

    # For VisA
    # CLASS_NAMES = ['capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum', 'candle']
    
    # For MVTec-AD
    CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                    'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    px = []
    im = []

    for class_name in CLASS_NAMES:

        localnet = model.localnet(1536)
        saved_dir = os.path.join('/path/to/glml/results', class_name, 'name_of_pretrained_model.pt')
        localnet.load_state_dict(torch.load(saved_dir)['net'])

        feature_path = '/path/to/DINO/'
        test_features = np.load(os.path.join(feature_path, class_name+'_test.npy')).squeeze()
        label_gt = np.load(os.path.join(feature_path, class_name+'_gt.npy'))
        mask_gt = np.load(os.path.join(feature_path, class_name+'_mask.npy')).squeeze()
        mask_gt = np.ceil(mask_gt)

        test_dataset = []
        for i in range(test_features.shape[0]):
            test_dataset.append([test_features[i], label_gt[i], mask_gt[i]])
        test_loader = DataLoader(test_dataset, batch_size=8, pin_memory=True)

        seg_map = []
        img_map = []

        label_gt = []
        mask_gt = []

        for x, y, mask in tqdm.tqdm(test_loader):
            with torch.no_grad():
                localnet = localnet.to(device)
                localnet.eval()

                x = x.to(device)
                y = y.detach().numpy()
                mask = mask.detach().numpy()
                _, score = localnet(x)
                score = score.detach().cpu().numpy()

                img_score = score.max(axis=-1)
                if x.shape[0] == 1:
                    img_score = np.array([img_score])

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

        print(class_name, '|', f'auroc: {auroc:.5f}, pxiel auroc: {pixel_auroc:.5f}')

        px.append(pixel_auroc)
        im.append(auroc)
    
    px = np.mean(np.array(px))
    im = np.mean(np.array(im))
    print('average|', f'auroc: {im:.5f}, pxiel auroc: {px:.5f}')

if __name__ == "__main__":
    main()