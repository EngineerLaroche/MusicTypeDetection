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

import math

from tensorflow.keras.callbacks import Callback

class BestWeightsRestorer(Callback):

    def __init__(self, monitor='val_loss', mode='min', verbose=0):

        self._monitor = monitor
        self._mode = mode
        self._verbose = verbose
        self._decision_func = min if mode == 'min' else max
        self._best = math.inf if mode == 'min' else - math.inf
        self._best_weights = None
        self._best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):

        self._best_epoch += 1

        current = logs[self._monitor]

        if self._best != self._decision_func(self._best, current):

            self._best = current
            self._best_weights = self.model.get_weights()

            if self._verbose == 2:
                print('A new best epoch has been encountered')

    def on_train_end(self, logs=None):

        if self._best_weights is not None:

            if self._verbose in [1, 2]:
                print('Restoring the best weights from epoch ' + str(self._best_epoch))

            self.model.set_weights(self._best_weights)