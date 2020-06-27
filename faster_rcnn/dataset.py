import pandas as pd
import numpy as np
import cv2
import os

import torch

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
    """
    Kaggle wheat detection challenge PyTorch Dataset.

    Adapted from: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
    """
    def __init__(self, dataframe, image_dir, transforms=None,
                 bbox_transforms=None, train=True, load_images=False):
        """
        load_images - load all images into memory to avoid latency of reloading them.
        """
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()

        self.images_loaded = load_images

        if load_images:
            self.images = []
            for image_id in self.image_ids:
                image = cv2.imread(os.path.join(image_dir,
                                                f'{image_id}.jpg'),
                                   cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
                image /= 255.0
                self.images.append(image)

        # index on image_id for some performance improvements
        self.df = dataframe.set_index('image_id').sort_index()

        self.image_dir = image_dir

        self.transforms = transforms
        self.bbox_transforms = bbox_transforms

        self.train = train

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        records = self.df.loc[[image_id]]

        if self.images_loaded:
            image = self.images[index]
        else:
            image = cv2.imread(os.path.join(self.image_dir,
                                            f'{image_id}.jpg'),
                               cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            image /= 255.0

        sample = {'image': image}

        if self.train:
            # temporary solution to mislabeling in the training data
            # ref: https://www.kaggle.com/c/global-wheat-detection/discussion/149032
            records['area'] = records['w'] * records['h']
            records = records[records['area'].between(150, 1100000)]

            boxes = records[['x', 'y', 'w', 'h']].values

            area = records['area']
            area = torch.as_tensor(area, dtype=torch.float32)

            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

            # there is only one class
            labels = torch.ones((records.shape[0],), dtype=torch.int64)

            # assume all instances are not crowd
            iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([index]),
                'area': area,
                'iscrowd': iscrowd
            }

            sample['bboxes'] = target['boxes']
            sample['labels'] = labels

        if self.transforms:
            sample = self.transforms(**sample)
            image = sample['image']

            if self.bbox_transforms and self.train:
                sample['bboxes'] = self.bbox_transforms(sample['bboxes'])
                target['boxes'] = torch.tensor(sample['bboxes'])

        if self.train:
            return image, target, image_id
        else:
            return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
