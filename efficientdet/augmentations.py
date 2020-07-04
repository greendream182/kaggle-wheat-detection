import cv2
import random
import torch

import numpy as np

from torchvision.transforms.functional import adjust_brightness, adjust_contrast,adjust_gamma, to_pil_image


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

        new_sample = {'img': torch.from_numpy(new_image).to(torch.float32), 'scale': scale}

        if 'annot' in sample:
            annots = sample['annot']
            annots[:, :4] *= scale
            new_sample['annot'] = torch.from_numpy(annots)

        return new_sample


class Flip(object):
    """Flip image (mirror)."""

    def __call__(self, sample, flip_x=0.5, flip_y=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            _, cols, _ = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        if np.random.rand() < flip_y:
            image, annots = sample['img'], sample['annot']
            image = image[::-1, :, :]

            _, cols, _ = image.shape

            y1 = annots[:, 1].copy()
            y2 = annots[:, 3].copy()

            y_tmp = y1.copy()

            annots[:, 1] = cols - y2
            annots[:, 3] = cols - y_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class GaussBlur(object):
    """ Multiple types of noise in one class. """

    def __init__(self, p=0.5, kernel_size=(5, 5), 
                 stdx=1, stdy=1):
        self.p = p
        self.kernel_size = kernel_size
        self.stdx = stdx
        self.stdy = stdy

    def __call__(self, sample):
        if np.random.rand() < self.p:
            sample['img'] = cv2.GaussBlur(sample['img'], self.kernel_size,
                                          self.stdx, self.stdy)
        return sample


class AdjustBrightness(object):

    def __init__(self, p=0.5, lb=0.7, ub=1.3):
        self.p = p
        self.lb = lb
        self.ub = ub

    def __call__(self, sample):
        if np.random.rand() < self.p:
            f = np.random.uniform(self.lb, self.ub)
            sample['img'] = np.array(adjust_brightness(to_pil_image(sample['img']), f))
        return sample


class AdjustContrast(object):

    def __init__(self, p=0.5, lb=0.7, ub=1.3):
        self.p = p
        self.lb = lb
        self.ub = ub

    def __call__(self, sample):
        if np.random.rand() < self.p:
            f = np.random.uniform(self.lb, self.ub)
            sample['img'] = np.array(adjust_contrast(to_pil_image(sample['img']), f))
        return sample


class AdjustGamma(object):

    def __init__(self, p=0.5, lb=0.7, ub=1.3):
        self.p = p
        self.lb = lb
        self.ub = ub

    def __call__(self, sample):
        if np.random.rand() < self.p:
            f = np.random.uniform(self.lb, self.ub)
            sample['img'] = np.array(adjust_gamma(to_pil_image(sample['img']), f))
        return sample


class RandomRotate(object):
    """ Randomly rotate 90, 180, or 270 degrees. """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.rand() < self.p:
            ymax = sample['img'].shape[0] - 1

            for _ in range(random.randint(1, 3)):
                # rotate 90 degrees up to 3 times
                sample['img'] = np.rot90(sample['img'])
                annot_new = sample['annot']
                annot_new[:, 0] = ymax - sample['annot'][:, 3]
                annot_new[:, 1] = sample['annot'][:, 0]
                annot_new[:, 2] = ymax - sample['annot'][:, 1]
                annot_new[:, 3] = sample['annot'][:, 2]
                sample['annot'] = annot_new
        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image = sample['img']
        sample['img'] = ((image.astype(np.float32) - self.mean) / self.std)

        return sample
