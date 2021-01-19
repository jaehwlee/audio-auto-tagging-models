import numpy as np
import math
import os
from data_loader.mtat_loader import DataLoader
from data_loader.train_loader import TrainLoader
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import (
    CSVLogger,
    ModelCheckpoint,
)
from model import resemul
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
import tensorflow as tf


def make_checkpoint():
    checkpoint_path = "checkpoint/cp.cpkt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    return checkpointer


def load_data(root="../dataset"):
    train_data = TrainLoader(root=root, split="train")
    valid_data = DataLoader(root=root, split="valid")
    test_data = DataLoader(root=root, split="test")
    return train_data, valid_data, test_data


def train():
    train_data, valid_data, test_data = load_data()

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = resemul()
        csv_logger = CSVLogger("./checkpoint/log.csv", append=True)
        checkpointer = make_checkpoint()

        for i in range(4):
            if i == 0:
                optimizer = Adam(learning_rate=1e-4)
                epoch = 60

            elif i == 1:
                optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
                epoch = 20
            elif i == 2:
                optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
                epoch = 20

            else:
                optimizer = SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)
                epoch = 100

            model.compile(
                loss="binary_crossentropy",
                optimizer=optimizer,
                metrics=[AUC(name="roc_auc"), AUC(name="pr_auc", curve="PR")],
            )
            model.fit(
                train_data,
                batch_size=16,
                epochs=epoch,
                callbacks=[checkpointer, csv_logger],
            )
            loss, roc_auc, pr_auc = model.evaluate(valid_data)
            print("roc_auc : {}, pr_auc : {}, loss : {}".format(roc_auc, pr_auc, loss))
            model.save("../models/model" + str(i) + ".h5")

        loss, roc_auc, pr_auc = model.evaluate(test_data)
        print("roc_auc : {}, pr_auc : {}, loss : {}".format(roc_auc, pr_auc, loss))


if __name__ == "__main__":
    train()
