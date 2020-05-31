"""
Some convenience functions.
"""
import cv2

import matplotlib.pyplot as plt
import numpy as np


def plot_image_and_bboxes(im, bboxes=[], colors=[], bw=2,
                          ax=None, figsize=(16, 16)):
    """
    im - ndarray (w, h, c)
    bboxes - list of bboxes (xmin, ymin, xmax, ymax)
    colors - list of colors for the bboxes (r, g, b)
    bw - width of bboxes
    """
    if ax is None:
        fig, ax = plt.suplots(1, 1, figsize=figsize)

    if not colors:
        colors = [(255, 0, 0) for i in range(len(bboxes))]
    elif isinstance(colors, tuple):
        colors = [colors for i in range(len(bboxes))]

    for box, color in zip(bboxes, colors):
        cv2.rectangle(im,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      color,
                      bw)

    ax.set_axis_off()
    ax.imshow(im)
    return fig


def gauss_noise_bbox(bbox, mu=0, sigma=1):
    """
    bbox - tuple (x, y, x+width, y+height)
    mu - mean
    sigma - std
    """
    return (i + np.random.normal(mu, sigma) for i in bbox)


def log_message(msg, logger, verbose=True):
    """ Log message, print if verbose. """
    logger.info(msg)
    if verbose:
        print(msg)


def format_prediction_string(boxes, scores):
    """
    Source: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-inference
    """
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)


class LossAverager:
    """
    Clean way to track loss over several epochs.
    Taken from: https://www.kaggle.com/pestipeti/pytorch-starter-fasterrcnn-train
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
