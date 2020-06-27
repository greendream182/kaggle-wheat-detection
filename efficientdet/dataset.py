import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
import cv2


def get_train_test_df(base_dir):
    """
    Returns dataframes with image_ids for train and test data.

    base_dir must have format:
    ├── train.csv
    ├── sample_sumbission.csv
    ├── test
    └── train
    """
    train_df = pd.read_csv(os.path.join(base_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(base_dir, 'sample_submission.csv'))

    bbox_cols = ['x', 'y', 'w', 'h']
    train_df = train_df.assign(**dict.fromkeys(bbox_cols, -1.))

    train_df.loc[:, bbox_cols] = train_df['bbox'].str.replace('[^0-9 .]',
                                                              '',
                                                              regex=True).str.split().tolist()
    train_df.loc[:, bbox_cols] = train_df.loc[:, bbox_cols].astype(np.float)

    return train_df, test_df


class WheatDataset(Dataset):

    def __init__(self, dataframe, image_dir, transform=None, train=True):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()

        # index on image_id for some performance improvements
        self.df = dataframe.set_index('image_id').sort_index()

        self.image_dir = image_dir

        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        if self.train:
            annot = self.load_annotations(idx)
            sample = {'img': img, 'annot': annot}
        else:

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_id = self.image_ids[image_index]
        img = cv2.imread(os.path.join(self.image_dir, f'{image_id}.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        image_id = self.image_ids[index]
        records = self.df.loc[[image_id]]

        # artificial class label (one class)
        records['category_id'] = 0

        # temporary solution to mislabeling in the training data
        # ref: https://www.kaggle.com/c/global-wheat-detection/discussion/149032
        records['area'] = records['w'] * records['h']
        records = records[records['area'].between(150, 1100000)]

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


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image = sample['img']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        sample = {'img': torch.from_numpy(new_image).to(torch.float32), 'scale': scale}

        if 'annot' in sample:
            annots = sample['annot']
            annots = annots[:, :4] *= scale
            sample['annot'] = torch.from_numpy(annots)

        return sample


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        sample['img'] = ((image.astype(np.float32) - self.mean) / self.std)

        return sample
