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

"""
Ce fichier contient les constantes globale.

Pour les chemins des fichiers, veuillez en faire des chemins relatifs au root du projet.
"""


from pathlib import Path
PROJECT_ROOT_PATH = str(Path(__file__).parent.parent) + '/'
DATA_PATH = PROJECT_ROOT_PATH + 'data/'
PROCESSED_DATA_PATH = DATA_PATH + 'processed/'
PROCESSED_IMG_PATH = PROCESSED_DATA_PATH + 'imgs/'
RAW_DATA_PATH = DATA_PATH + 'raw/'
RAW_TAGGED_FEATURE_SET_PATH = RAW_DATA_PATH + 'music/tagged_feature_sets/'
LOGS_PATH = PROJECT_ROOT_PATH + 'logs/'
MODELS_PATH = PROJECT_ROOT_PATH + 'models/'
