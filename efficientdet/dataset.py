import cv2
import os
import random
import torch

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def get_train_test_df(base_dir):
    """
    Returns dataframes with image_ids for train and test data.

    base_dir must have format:
    ├── train.csv
    ├── sample_sumbission.csv
    ├── test
    └── train
    """
    train_df = pd.read_csv(os.path.join(base_dir, 'train_df_fixed.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'test.csv'))

    def extract_bboxes(df):
        bbox_cols = ['x', 'y', 'w', 'h']
        df = df.assign(**dict.fromkeys(bbox_cols, np.nan))

        df.loc[:, bbox_cols] = df['bbox'].str.replace('[^0-9 .]',
                                                      '',
                                                      regex=True).str.split().tolist()
        df.loc[:, bbox_cols] = df.loc[:, bbox_cols].astype(np.float32)
        return df

    train_df = extract_bboxes(train_df)
    test_df = extract_bboxes(test_df)

    return train_df, test_df


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None, 
                 mixup=False, train=True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()

        # index on image_id for some performance improvements
        self.df = dataframe.set_index('image_id').sort_index()

        self.image_dir = image_dir

        self.transform = transform
        self.mixup = mixup
        self.train = train


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        if not self.mixup:
            return self._getitem(idx)
        else:
            return self._getmixup(idx)

    def _getitem(self, idx):
        img = self.load_image(idx)
        if self.train:
            annot = self.load_annotations(idx)
            sample = {'img': img, 'annot': annot}
        else:
            sample = {'img': img}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def _getmixup(self, idx):
        """
        Mixup in this case represents swapping a horzontal or vertical cut of the images.
        """
        if not self.train:
            raise RuntimeWarning('Can not use mixup when in test mode.')
            return self._getitem(idx)

        img = self.load_image(idx)
        annot = self.load_annotations(idx)

        randidx = random.randint(0, len(self.image_ids) - 1)
        if randidx == idx:
            return sample

        randimg = self.load_image(randidx)
        randannot = self.load_annotations(randidx)

        # mixup must swap at least 1/8th of the image, at most 2/3
        min_swp = int(img.shape[0] * 0.125)
        max_swp = int(img.shape[0] * 0.66)
        cutsz = random.randint(min_swp, max_swp)
        
        if np.random.rand() <= 0.5:
            # horizontal cut
            img[:cutsz, :, :] = randimg[:cutsz, :, :]

            randannot = randannot[(randannot[:, 1] + 45) < cutsz] # ymin too high
            randannot[:, 3] = np.where(randannot[:, 3] >= cutsz, cutsz-1, randannot[:, 3])
            
            annot = annot[(annot[:, 3] + 45) > cutsz] # drop annot w. low ymax
            annot[:, 1] = np.where(annot[:, 1] < cutsz, cutsz, annot[:, 1])

            annot = np.concatenate([annot, randannot])
        else:
            # vertical cut
            img[:, :cutsz, :] = randimg[:, :cutsz, :]

            randannot = randannot[(randannot[:, 0] + 45) < cutsz] # xmin too high
            randannot[:, 2] = np.where(randannot[:, 2] >= cutsz, cutsz-1, randannot[:, 2])
            
            annot = annot[(annot[:, 2] + 45) > cutsz] # drop annot w. low xmax
            annot[:, 0] = np.where(annot[:, 0] < cutsz, cutsz, annot[:, 0])

            annot = np.concatenate([annot, randannot])

        # hack-fix
        # drop annotations with width/height 25 or fewer pixels
        annot = annot[annot[:, 2] - annot[:, 0] > 25]
        annot = annot[annot[:, 3] - annot[:, 1] > 25]
        
        return {'img': img, 'annot': annot}
 
    def load_image(self, image_index):
        image_id = self.image_ids[image_index]
        img = cv2.imread(os.path.join(self.image_dir, f'{image_id}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        image_id = self.image_ids[image_index]
        records = self.df.loc[[image_id]]

        # artificial class label (one class)
        records['category_id'] = 0

        annotations = records[['x', 'y', 'w', 'h', 'category_id']].values

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}

if __name__ == '__main__':
    #for debugging
    import os
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

    data_dir = os.path.join(base_dir, 'data')
    train_imgs_dir = os.path.join(data_dir, 'train')
    test_imgs_dir = os.path.join(data_dir, 'test')

    train_df, test_df = get_train_test_df(data_dir)

    immean = [0.485, 0.456, 0.406]
    imstd = [0.229, 0.224, 0.225]

    from augmentations import Normalizer, Flip, Resizer, GaussBlur, AdjustBrightness, AdjustContrast, AdjustGamma, RandomRotate
    from torchvision import transforms
    train_transform = transforms.Compose([Normalizer(mean=immean, std=imstd),
                                          Flip(),
                                          GaussBlur(p=0.5),
                                          AdjustContrast(p=0.3),
                                          AdjustBrightness(p=0.3),
                                          AdjustGamma(p=0.3),
                                          RandomRotate(),
                                          Resizer(1280)])
    test_transform = transforms.Compose([Normalizer(mean=immean, std=imstd),
                                         Resizer(1280)])

    train_dataset = WheatDataset(train_df, train_imgs_dir,
                                 train_transform, mixup=True)

    test_dataset = WheatDataset(test_df, test_imgs_dir,
                                test_transform, mixup=False)

    import cv2
    import matplotlib.pyplot as plt

    def plot_image_and_bboxes(im, bboxes=[], colors=[], bw=2,
                          ax=None, figsize=(16, 16)):
        """
        im - ndarray (w, h, c)
        bboxes - list of bboxes (xmin, ymin, xmax, ymax)
        colors - list of colors for the bboxes (r, g, b)
        bw - width of bboxes
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if not colors:
            colors = [(255, 0, 0) for i in range(len(bboxes))]
        elif isinstance(colors, tuple):
            colors = [colors for i in range(len(bboxes))]

        for box, color in zip(bboxes, colors):
            cv2.rectangle(im,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color,
                        bw)

        ax.set_axis_off()
        ax.imshow(im)
        return fig

    import IPython; IPython.embed(); exit(1)
