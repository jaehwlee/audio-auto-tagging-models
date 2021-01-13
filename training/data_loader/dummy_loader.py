import numpy as np
import math
from tensorflow.keras.utils import Sequence
import os


class Dataloader(Sequence):
    def __init__(self, root, batch_size=16, input_length=59049):
        self.root = root
        self.input_length = input_length
        self.batch_size = batch_size
        self.on_epoch_end()

    def __getitem__(self, idx):
        a = np.zeros((16, 59049),)
        b = np.zeros((16, 50))
        return a, b

    def on_epoch_end(self):
        self.indices = np.arange(len(self.fl))

    def __len__(self):
        return math.ceil(len(self.fl) / self.batch_size)
