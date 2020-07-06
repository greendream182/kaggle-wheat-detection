"""
EfficientDet training script.

Adapted from https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch
"""
import datetime
import os
import torch
import traceback

import numpy as np

from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm

from efficientdet.backbone import EfficientDetBackbone
from efficientdet.dataset import get_train_test_df, WheatDataset, collater
from efficientdet.augmentations import Normalizer, Flip, Resizer, GaussBlur, AdjustBrightness, AdjustContrast, AdjustGamma, RandomRotate

from efficientdet.efficientdet.loss import FocalLoss
from efficientdet.utils.sync_batchnorm import patch_replication_callback
from efficientdet.utils.utils import replace_w_sync_bn, CustomDataParallel, init_weights

# Optional mixed precision idea taken fromhttps://github.com/ultralytics/yolov5/blob/master/train.py 
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed


class ModelWithLoss(nn.Module):
    def __init__(self, model, debug=False):
        super().__init__()
        self.criterion = FocalLoss()
        self.model = model
        self.debug = debug

    def forward(self, imgs, annotations, obj_list=None):
        _, regression, classification, anchors = self.model(imgs)
        if self.debug:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations,
                                                imgs=imgs, obj_list=obj_list)
        else:
            cls_loss, reg_loss = self.criterion(classification, regression, anchors, annotations)
        return cls_loss, reg_loss


def save_checkpoint(model, name, path):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(path, name))


def train(base_dir, batch_size=8, lr=10e-4, num_epochs=20, num_workers=12, version=5, weights_path=None, head_only=False,
          num_gpus=1, optim='adamw', seed=15501, save_interval=2000, out_dir='.', debug=False):
    """[summary]

    Parameters
    ----------
    base_dir : str
        Directory to kaggle wheat challenge type directory for training.
    batch_size : int, optional
        batch size, by default 8
    lr : float, optional
        learning rate, by default 10e-4
    num_epochs : int, optional
        number of epochs to train for, by default 20
    num_workers : int, optional
        num_workers for dataloader, by default 12
    version : int, optional
        efficientdet model version, by default 5
    weights_path : str, optional
        path to pretrained model, by default None
    head_only : bool, optional
        train only head, by default False
    num_gpus : int, optional
        number of GPUs to train with, by default 1
    seed : int, optional
        random seed, by default 15501
    save_interval : int, optional
        checkpoint model every save_interval steps, by default 2000
    out_dir : str, optional
        output directory for logs and checkpoints, by default '.'
    debug : bool, optional
        debug flag, by default False
    """
    data_dir = os.path.join(base_dir, 'data')
    train_imgs_dir = os.path.join(data_dir, 'train')
    test_imgs_dir = os.path.join(data_dir, 'test')

    if num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)

    saved_path = out_dir
    log_path = os.path.join(out_dir, 'tensorboard')
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(saved_path, exist_ok=True)

    train_df, test_df = get_train_test_df(data_dir)

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    immean = [0.485, 0.456, 0.406]
    imstd = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([Normalizer(mean=immean, std=imstd),
                                          Flip(),
                                          GaussBlur(p=0.5),
                                          AdjustContrast(p=0.3),
                                          AdjustBrightness(p=0.3),
                                          AdjustGamma(p=0.3),
                                          RandomRotate(),
                                          Resizer(input_sizes[version])])
    test_transform = transforms.Compose([Normalizer(mean=immean, std=imstd),
                                         Resizer(input_sizes[version])])

    train_dataset = WheatDataset(train_df, train_imgs_dir,
                                 train_transform, mixup=True)

    test_dataset = WheatDataset(test_df, test_imgs_dir,
                                test_transform, mixup=False)

    training_params = {'batch_size': batch_size,
                       'shuffle': True,
                       'drop_last': False,
                       'collate_fn': collater,
                       'num_workers': num_workers}

    test_params = {'batch_size': batch_size,
                   'shuffle': False,
                   'drop_last': False,
                   'collate_fn': collater,
                   'num_workers': num_workers}

    training_generator = DataLoader(train_dataset, **training_params)
    test_generator = DataLoader(test_dataset, **test_params)

    anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
    anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'
    model = EfficientDetBackbone(num_classes=1, compound_coef=version,
                                 ratios=eval(anchors_ratios), scales=eval(anchors_scales))

    # load last weights
    if weights_path is not None:
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] froze backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if num_gpus > 1 and batch_size // num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=debug)

    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            model = CustomDataParallel(model, num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, nesterov=True)

    # Automatic mixed precision. Should enable large batches.
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    print('[INFO] Starting training.')

    try:
        for epoch in range(num_epochs):
            # last_epoch = step // num_iter_per_epoch
            # if epoch < last_epoch:
            #     continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=['wheat'])
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % save_interval == save_interval and step > 0:
                        save_checkpoint(model, f'efficientdet-d{version}_{epoch}_{step}.pth', saved_path)
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            ## Evaluation done here.
            ## Note: certain things are hardcoded here, done to save time and make sure things run.
            if epoch % 1 == 0: # Eval on each epoch (training is slow)
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(test_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=['wheat'])
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                # hardcoded:
                es_min_delta = 0.1
                if loss + es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'efficientdet-d{version}_{epoch}_{step}.pth', saved_path)

                model.train()
                           
                # Early stopping
                # hardcoded:
                es_patience = 5
                if epoch - best_epoch > es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break

    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{version}_{epoch}_{step}.pth', saved_path)
        writer.close()
    writer.close()
