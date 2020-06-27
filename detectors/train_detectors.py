"""
Training script for DetectoRS.

Pretrained model downloaded from: https://github.com/joe-siyuan-qiao/DetectoRS
"""
import logging
import torch

import numpy as np
import pandas as pd


def train(base_dir, n_splits=5, n_epochs=40, batch_size=16,
          train_folds=None, model_name='detectors-tuned',
          eval_per_n_epochs=10, seed=15501, verbose=True):
    """Train the DetectoRS model.

    Parameters
    ----------
    base_dir : str
        Path to base data directory.
    n_splits : int, optional
        Number of cross validation splits, by default 5
    n_epochs : int, optional
        Number of epochs to train for, by default 40
    batch_size : int, optional
        Batch size, by default 16
    train_folds : list, optional
        List of folds to train on this run, by default None.
        Trains all folds if None.
        Example: [2, 3] will train folds 2 and 3.
        Need to pass same seed to guarantee consistentcy.
    model_name : str, optional
        Save trained model with name, by default 'detectors-tuned'
    eval_per_n_epochs : int, optional
        Evaluate model on val set every n epochs, by default 10
    seed : int, optional
        Random seed, by default 15501
    verbose : bool, optional
        Prints updates while running, by default True
    """
    raise NotImplementedError