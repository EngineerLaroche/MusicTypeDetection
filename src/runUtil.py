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
import json
import os
from typing import List, Dict, Any

import numpy as np

import src.constants as constants


class NumpyDtypeJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        else:
            return json.JSONEncoder.default(self, obj)


def resume_history(history):
    """
    Résume l'historique d'exécution d'un modèle Keras en extractant les métriques de la meilleur époque.

    Args:
        history: 
    Returns: Un OrderedDict contenant les métriques de la meilleur époque.

    """

    resume = []

    metrics = np.array([
        'loss',
        'val_loss',
        'accuracy',
        'val_accuracy'
    ])

    best_run_i = np.argmin(history['val_loss'])
    resume.append(('best_epoch', best_run_i + 1))

    for metric in metrics:
        if metric in history:
            resume.append((metric, history[metric][best_run_i]))

    return OrderedDict(resume)


def save_run_history(run_name: str, history: np.ndarray):

    history_save_path = constants.LOGS_PATH + run_name + '_history'

    np.save(history_save_path, history)


def save_run_results(run_name: str, run_results):

    file_path = _run_results_file_path(run_name)

    with open(file_path, 'w') as file:
        json.dump(run_results, file, cls=NumpyDtypeJSONEncoder)


def load_run_results(run_name: str):

    file_path = _run_results_file_path(run_name)

    with open(file_path, 'r') as file:
        run_results = json.load(file, object_pairs_hook=OrderedDict)

    return run_results


def _run_results_file_path(run_name: str) -> str:

    return constants.LOGS_PATH + run_name + '_run_results.json'


def save_run_config(run_name: str, run_config):

    file_path = constants.LOGS_PATH + run_name + '_run_config.json'

    with open(file_path, 'w') as file:
        json.dump(run_config, file, cls=NumpyDtypeJSONEncoder)
