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

import joblib
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

from src import constants

def _load_feature_set(path):
    """
    Charge les données d'un ensemble de primitives.

    Args:
        path (str): Le chemin du fichier contenant les données de l'ensemble de primitives

    Returns:
        Un tuple contennant une liste des données de chaque exemples, ainsi qu'une liste contenant l'id de chaque
        exemples .
        ([dataset], [ids])
    """

    ftrs = np.array(pd.read_csv(path, header=None).values[: ,2:-1])
    ids = np.array(pd.read_csv(path, header=None).values[: ,1])

    return ftrs, ids

def _load_feature_sets(paths):
    """
    Charge les données de tous les ensembles de primitives dont le chemin du fichier est passé en paramêtre.

    Args:
        path (list(str)): Le chemin des fichiers contenant les données des ensembles de primitives

    Returns:
        Un tuple contennant une liste des ensembles de données de chaque primitives , ainsi qu'une liste contenant l'id
        de chaque exemples.
        ([ [feature_set_1], [feature_set_2], ... ], [ids])
    """

    first_set, ids = _load_feature_set(paths[0])

    ftrs_sets_raw = [first_set]

    for ftrs_set_pth in  paths[1:]:
        ftrs_set_raw = _load_feature_set(ftrs_set_pth)[0]
        ftrs_sets_raw.append(ftrs_set_raw)

    return ftrs_sets_raw, ids

def _load_output_mapping():

    with open(constants.PROJECT_ROOT_PATH+ 'output_mapping/output_mapping.json', 'r') as file:
        output_mapping = json.load(file)

    return output_mapping

def _scale_ftrs_sets(ftrs_sets_raw):
    """
    Effectue un standard scaling sur les données d'entrées.

    Args:
        ftrs_sets_raw (List(List(any))): Liste contenant les ensembles de données d'entrées.
    """

    proc_ftrs_sets = []

    for ftrs_set_raw in ftrs_sets_raw:

        scaler = StandardScaler()
        ftrs_scld = scaler.fit_transform(ftrs_set_raw)

        proc_ftrs_sets.append(ftrs_scld)

    return proc_ftrs_sets


output_file_path = constants.PROJECT_ROOT_PATH + 'predictions/predictions3.csv'
input_fils_paths = [
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-marsyas_test_new_nolabels/msd-marsyas_test_new_nolabels.csv',
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv',
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-jmirderivatives_test_nolabels/msd-jmirderivatives_test_nolabels.csv',
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-jmirmfccs_test_nolabels/msd-jmirmfccs_test_nolabels.csv',
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-jmirderivatives_test_nolabels/msd-jmirderivatives_test_nolabels.csv',
    constants.PROJECT_ROOT_PATH + 'data/raw/music/untagged_feature_sets/msd-ssd_test_nolabels/msd-ssd_test_nolabels.csv',
]
ftrs_sets_raw, ids = _load_feature_sets(input_fils_paths)
ftrs_sets = _scale_ftrs_sets(ftrs_sets_raw)

ens_model = load_model(constants.MODELS_PATH + 'ensemble_five_final_4.h5')
svm_derivs_model = joblib.load(constants.MODELS_PATH + 'svm_final_1.joblib')
rf_ssd_model = pickle.load(open(constants.MODELS_PATH + 'final_random_forest_2.pickle', 'rb'))

output_mapping = _load_output_mapping()

ftrs_sets[4] = svm_derivs_model.predict_proba(ftrs_sets[4])
ftrs_sets[5] = rf_ssd_model.predict_proba(ftrs_sets[5])

predictions_prob = ens_model.predict(x=ftrs_sets, batch_size=2000)

observations_predictions = []

for i, id in enumerate(ids):

    observation_predictions = predictions_prob[i]
    best_class_index = np.argmax(observation_predictions)
    label = output_mapping[best_class_index]

    observations_predictions.append((id, label))

results = pd.DataFrame.from_records(
    data=observations_predictions,
    columns=['id', 'genre']
)

results.to_csv(path_or_buf=output_file_path, header=True, index=False)
