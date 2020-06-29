#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # 4 — Développement d’un système intelligent

Students :
    Alexandre Laroche - LARA12078907
    Marc-Antoine Charland - CHAM16059609
    Jonathan Croteau-Dicaire - CROJ10109402

Group :
    GTI770-É19-02
"""

from collections import OrderedDict
import statistics
import time

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import StratifiedKFold


def cross_validate(x, y, y_label_encoded, n_splits, train_func, **kwargs):

    y_true = []
    y_predicted = []
    train_time = 0
    histories = []
    results = []

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=False)
    fold = 1

    for train_index, val_index in kfold.split(x, y_label_encoded):

        y_true.extend(y_label_encoded[val_index])

        kwargs['fold'] = fold

        start_train_time = time.perf_counter()

        result = train_func(x[train_index], y[train_index], x[val_index], y[val_index], **kwargs)

        train_time += time.perf_counter() - start_train_time

        histories.append(result['history'])
        y_predicted.extend(result['predictions'])
        fold += 1

    results.append(('avg_val_f1_micro', f1_score(y_true=y_true, y_pred=y_predicted, average='micro')))
    results.append(('avg_val_f1_micro', f1_score(y_true=y_true, y_pred=y_predicted, average='micro')))
    results.append(('avg_val_f1_macro', f1_score(y_true=y_true, y_pred=y_predicted, average='macro')))
    results.append(('avg_val_accuracy', accuracy_score(y_true=y_true, y_pred=y_predicted)))
    results.append(('avg_batch_val_accuracy', _avg_metric('val_accuracy', histories)))
    results.append(('avg_accuracy', _avg_metric('accuracy', histories)))
    results.append(('avg_val_loss', _avg_metric('val_loss', histories)))
    results.append(('avg_loss', _avg_metric('loss', histories)))
    results.append(('train_time', train_time))

    return OrderedDict(results)

def _avg_metric(metric, histories):
    """
    Compte la moyenne d'un metric des meilleurs epochs des history
    :param histories: Une liste d'histories pour lesquel ont veut la meilleur metric moyenne
    :return: La metric moyen
    """

    metrics = []

    for history in histories:
        min_val_loss = min(history['val_loss'])
        best_epoch_index = history['val_loss'].index(min_val_loss)

        metrics.append(history[metric][best_epoch_index])

    return np.mean(metrics)
