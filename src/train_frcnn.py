"""
Train script for a faster-rcnn baseline.

Potential improvements:
    Mixed-examples data augmentation (may need to change the Dataset for this):
        https://arxiv.org/pdf/1805.11272.pdf
        https://github.com/ufownl/global-wheat-detection/blob/master/dataset.py#L149
    Tuning hyperparameters
"""
import logging
import os
import torch
import torchvision

import numpy as np
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.model_selection import GroupKFold

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torch.utils.data import DataLoader

from .data import get_train_test_df, WheatDataset
from .eval import calculate_image_precision_by_threshold
from .utils import gauss_noise_bboxes, LossAverager, log_message


# Data augmentations
def get_train_transform():
    transforms = [
        A.Flip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomSizedCrop(min_max_height=(600, 950),
                          height=1024,
                          width=1024,
                          p=0.5),
        # A.RandomCrop(800, 800, p=0.7),
        A.GaussNoise(var_limit=(0.05, 0.15), p=0.7),
        A.RandomBrightnessContrast(),
        # A.Normalize(),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.25),
        A.Resize(800, 800),
        ToTensorV2(p=1.0)
    ]
    bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    }
    return A.Compose(transforms,
                     bbox_params=bbox_params)


def get_valid_transform():
    transforms = [
        # A.Normalize(),
        A.Resize(800, 800),
        ToTensorV2(p=1.0)
    ]
    bbox_params = {
        'format': 'pascal_voc',
        'label_fields': ['labels']
    }
    return A.Compose(transforms,
                     bbox_params=bbox_params)


def get_test_transform():
    transforms = [
        # A.Normalize(),
        A.Resize(800, 800),
        ToTensorV2(p=1.0)
    ]
    return A.Compose(transforms)


def bbox_transform(bboxes):
    """
    Transformations to be done to the bboxes.
    """
    bboxes = gauss_noise_bboxes(bboxes, sigma=2)

    return bboxes


def collate_fn(batch):
    """ https://discuss.pytorch.org/t/how-to-use-collate-fn/27181 """
    return tuple(zip(*batch))


def train(base_dir, n_splits=5, n_epochs=40, batch_size=16,
          train_folds=None, model_name='faster-rcnn-baseline',
          eval_per_n_epochs=10, seed=15501, verbose=True):
    """
    Train frcnn baseline.
    Largely inspired by: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train

    train_splits expects a list/tuple of ints. If train_splits,
    only train the specified splits.
    """
    np.random.seed(seed)

    data_dir = os.path.join(base_dir, 'data')
    train_imgs_dir = os.path.join(data_dir, 'train')
    test_imgs_dir = os.path.join(data_dir, 'test')
    models_out_dir = os.path.join(base_dir, 'artifacts',
                                  model_name, 'models')
    os.makedirs(models_out_dir, exist_ok=True)
    preds_out_dir = os.path.join(base_dir, 'artifacts',
                                 model_name, 'predictions')
    os.makedirs(preds_out_dir, exist_ok=True)
    log_file = os.path.join(base_dir, 'artifacts', model_name, 'train.log')
    open(log_file, 'a').close() # create empty file.

    logger = logging.getLogger(model_name)
    logger.addHandler(logging.FileHandler(log_file))

    train_df, test_df = get_train_test_df(data_dir)

    kf = GroupKFold(n_splits)
    split = kf.split(X=train_df[['image_id']],
                     y=train_df[['x', 'y', 'w', 'h']],
                     groups=train_df['image_id'])

    if isinstance(train_folds, (list, tuple)):
        split = [fold for i, fold in enumerate(split) if i in train_folds]
        info = f'Training only on folds {train_folds}.'
        log_message(log_message, logger, verbose)
    else:
        train_folds = range(n_splits)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for split_n, (train_idx, val_idx) in zip(train_folds, split):
        info = f'Training fold {split_n} beginning.'
        log_message(info, logger, verbose)

        train = train_df.iloc[train_idx].copy()
        val = train_df.iloc[val_idx].copy()

        train_dataset = WheatDataset(train, train_imgs_dir,
                                     get_train_transform(),
                                     bbox_transforms=bbox_transform)
        val_dataset = WheatDataset(val, train_imgs_dir,
                                   get_valid_transform(),
                                   return_image_id=True)

        test_dataset = WheatDataset(test_df, test_imgs_dir,
                                    get_test_transform(),
                                    train=False)

        # load pretrained faster-rcnn with resnet50 backbone
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # update pre-trained head
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                          num_classes)

        train_data_loader = DataLoader(train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4,
                                       collate_fn=collate_fn)

        val_data_loader = DataLoader(val_dataset,
                                     batch_size=8,
                                     shuffle=False,
                                     num_workers=4,
                                     collate_fn=collate_fn)

        test_data_loader = DataLoader(test_dataset,
                                      batch_size=4,
                                      shuffle=False,
                                      num_workers=4,
                                      drop_last=False,
                                      collate_fn=collate_fn)

        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.95, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        loss_hist = LossAverager()

        for epoch in range(n_epochs):
            info = f'Training epoch #{epoch}.'
            log_message(info, logger, verbose)

            loss_hist.reset()

            for it, (images, targets) in enumerate(train_data_loader):
                images = list(image.to(device) for image in images)
                targets = [{k: v.long().to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()

                loss_hist.send(loss_value)

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                if it+1 % 20 == 0:
                    info = f'Iteration #{it} loss: {loss_value}'
                    log_message(info, logger, verbose)

            lr_scheduler.step()

            info = f'Epoch #{epoch} loss: {loss_hist.value}'
            log_message(info, logger, verbose)

            if epoch+1 % eval_per_n_epochs == 0:
                # may want to add this to eval.py... somehow?
                thresholds = np.linspace(0.5, 0.75, 6)
                precisions_by_thresh = []

                for images, targets, _ in val_data_loader:
                    images = list(image.to(device) for image in images)
                    outputs = model(images)

                    for targ, out in zip(targets, outputs):
                        gt = targ['boxes'].cpu().numpy().astype(np.int32)
                        scores = out['scores'].data.cpu().numpy()
                        # predictions ordered by confidence
                        preds = out['boxes'].data.cpu().numpy()[np.argsort(scores)]
                        ap_by_thresh = calculate_image_precision_by_threshold(gt,
                                                                              preds,
                                                                              thresholds=thresholds,
                                                                              form='pascal_voc')
                        precisions_by_thresh.append(ap_by_thresh)

                mean_precisions_by_thresh = np.array(precisions_by_thresh).mean(axis=0)
                mean_ap = mean_precisions_by_thresh.mean()

                for thresh, ap in zip(thresholds, mean_precisions_by_thresh):
                    info = f'Epoch #{epoch} - AP at IOU threshold {thresh}: {ap}.'
                    log_message(info, logger, verbose)

                info = f'Epoch #{epoch} - Mean AP across all thresholds: {mean_ap}.'
                log_message(info, logger, verbose)

        # save model.
        torch.save(model.state_dict(), os.path.join(models_out_dir,
                                                    f'trained_fold_{split_n}.pth'))

        detection_threshold = 0.1
        res = []

        for images, _, image_ids in val_data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for output, image_id in zip(outputs, image_ids):
                boxes = output['boxes'].data.cpu().numpy()
                scores = output['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]

                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                for out in np.hstack([boxes, scores]):
                    res.append([image_id] + list(out))

        df_res = pd.DataFrame(res, columns=['image_id', 'x', 'y', 'w', 'h', 'score'])
        df_res.to_csv(os.path.join(preds_out_dir, f'oof_pred_fold_{split_n}.csv'), index=False)

        detection_threshold = 0.1
        res = []

        for images, image_ids in test_data_loader:
            images = list(image.to(device) for image in images)
            outputs = model(images)

            for output, image_id in zip(outputs, image_ids):
                boxes = output['boxes'].data.cpu().numpy()
                scores = output['scores'].data.cpu().numpy()

                boxes = boxes[scores >= detection_threshold].astype(np.int32)
                scores = scores[scores >= detection_threshold]

                boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                for out in np.hstack([boxes, scores]):
                    res.append([image_id] + list(out))

        df_res = pd.DataFrame(res, columns=['image_id', 'x', 'y', 'w', 'h', 'score'])
        df_res.to_csv(os.path.join(preds_out_dir, f'test_pred_fold_{split_n}.csv'), index=False)
