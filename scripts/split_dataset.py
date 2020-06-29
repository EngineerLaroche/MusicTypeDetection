
import itertools
import os.path as path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import constants

def load_feature_set(name):
    file_path = path.join(constants.RAW_TAGGED_FEATURE_SET_PATH, 'msd-' + name + '/msd-' + name + '.csv')

    whole = np.array(pd.read_csv(file_path, header=None))
    lbls = np.array(pd.read_csv(file_path, header=None).values[:, -1])

    return whole, lbls


def unglue_feature_sets(glued_sets, sets_stop_index):
    sets = []

    stop_index = sets_stop_index[0]
    sets.append(glued_sets[:, :stop_index])

    i = 0
    while i < len(sets_stop_index) - 1:
        start_index = sets_stop_index[i]
        stop_index = sets_stop_index[i + 1]

        sets.append(glued_sets[:, start_index:stop_index])

        i += 1

    return sets


mary, lbls = load_feature_set('marsyas_dev_new')
ssd = load_feature_set('ssd_dev')[0]
derivs = load_feature_set('jmirderivatives_dev')[0]

ftrs_sets = [mary, ssd, derivs]

sets_stop_index = list(itertools.accumulate([ftrs_set.shape[1] for ftrs_set in ftrs_sets]))

print(sets_stop_index)

glued_sets = np.hstack(ftrs_sets)

base_gsets, meta_gsets = train_test_split(glued_sets, test_size=0.2, shuffle=True, stratify=lbls)

base_sets = unglue_feature_sets(base_gsets, sets_stop_index)
meta_sets = unglue_feature_sets(meta_gsets, sets_stop_index)

mary_base = pd.DataFrame(base_sets[0])
ssd_base = pd.DataFrame(base_sets[1])
derivs_base = pd.DataFrame(base_sets[2])

mary_meta = pd.DataFrame(meta_sets[0])
ssd_meta = pd.DataFrame(meta_sets[1])
derivs_meta = pd.DataFrame(meta_sets[2])

mary_base.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'marsyas_base_split.csv'), header=False, index=False)
ssd_base.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'ssd_base_split.csv'), header=False, index=False)
derivs_base.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'jmirderivatives_base_split.csv'), header=False, index=False)

mary_meta.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'marsyas_meta_split.csv'), header=False, index=False)
ssd_meta.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'ssd_meta_split.csv'), header=False, index=False)
derivs_meta.to_csv(path_or_buf=path.join(constants.DATA_PATH, 'jmirderivatives_meta_split.csv'), header=False, index=False)