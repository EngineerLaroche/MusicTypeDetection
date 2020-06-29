import itertools
import os.path as path
import numpy as np
import pandas as pd

from src import constants

def load_feature_set(name):
    file_path = path.join(constants.RAW_TAGGED_FEATURE_SET_PATH, 'msd-' + name + '/msd-' + name + '.csv')

    whole = np.array(pd.read_csv(file_path, header=None))

    return whole

base_file_path = path.join(constants.DATA_PATH, 'marsyas_base_split.csv')
meta_file_path = path.join(constants.DATA_PATH, 'marsyas_meta_split.csv')

ids_base = np.array(pd.read_csv(base_file_path, header=None).values[:, 0])
ids_meta = np.array(pd.read_csv(meta_file_path, header=None).values[:, 0])

dataset = load_feature_set('jmirmfccs_dev')

base_set = []
meta_set = []

for id in ids_base:
    i = np.where(dataset[:, 0] == id)
    base_set.append(dataset[i][0])

for id in ids_meta:
    i = np.where(dataset[:, 0] == id)
    meta_set.append(dataset[i][0])

base_set = pd.DataFrame(base_set)
meta_set = pd.DataFrame(meta_set)

base_set.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'jmirmfccs_base_split.csv'), header=False, index=False)
meta_set.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'jmirmfccs_meta_split.csv'), header=False, index=False)