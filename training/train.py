import numpy as np
import math
import os
from data_loader.mtat_loader import DataLoader
from data_loader.train_loader import TrainLoader
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    CSVLogger,
    LearningRateScheduler,
    EarlyStopping,
    ModelCheckpoint,
)
# from keras.models import Model, load_model
from model import music_sincnet
import argparse
from sklearn import metrics
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers.schedules import InverseTimeDecay
import tensorflow as tf

parser = argparse.ArgumentParser(description="music-sincnet")
parser.add_argument("--data-dir", type=str, default="./dataset", metavar="PATH")
parser.add_argument(
    "--train-dir",
    type=str,
    default="./log",
    metavar="PATH",
    help="Directory where to write event logs and checkpoints.",
)

parser.add_argument(
    "--block",
    type=str,
    default="se",
    choices=["se", "rese", "res", "basic"],
    help="Block to build a model: {se|rese|res|basic} (default: se).",
)
parser.add_argument(
    "--no-multi", action="store_true", help="Disables multi-level feature aggregation."
)
parser.add_argument(
    "--alpha", type=int, default=16, metavar="A", help="Amplifying ratio of SE block."
)

parser.add_argument(
    "--batch-size", type=int, default=23, metavar="N", help="Mini-batch size."
)
parser.add_argument(
    "--momentum", type=float, default=0.9, metavar="M", help="Momentum for SGD."
)
parser.add_argument(
    "--lr", type=float, default=0.01, metavar="LR", help="Learning rate."
)
parser.add_argument(
    "--lr-decay",
    type=float,
    default=0.2,
    metavar="DC",
    help="Learning rate decay rate.",
)

parser.add_argument(
    "--dropout", type=float, default=0.5, metavar="DO", help="Dropout rate."
)
parser.add_argument(
    "--weight-decay", type=float, default=0.0, metavar="WD", help="Weight decay."
)

parser.add_argument(
    "--initial-stage", type=int, default=0, metavar="N", help="Stage to start training."
)
parser.add_argument(
    "--patience",
    type=int,
    default=2,
    metavar="N",
    help="Stop training stage after #patiences.",
)
parser.add_argument(
    "--num-lr-decays",
    type=int,
    default=5,
    metavar="N",
    help="Number of learning rate decays.",
)

parser.add_argument(
    "--num-audios-per-shard",
    type=int,
    default=100,
    metavar="N",
    help="Number of audios per shard.",
)
parser.add_argument(
    "--num-segments-per-audio",
    type=int,
    default=10,
    metavar="N",
    help="Number of segments per audio.",
)
parser.add_argument(
    "--num-read-threads",
    type=int,
    default=8,
    metavar="N",
    help="Number of TFRecord readers.",
)

args = parser.parse_args()
# initial_lr = tf.Variable(0.01)

# def auc(y_true, y_pred):
#    auc = AUC(y_true, y_pred)[1]
#    K.get_session().run(tf.local_variables_initializer())
#    return auc


def load_data():
    x_train = np.load("../dataset/mtat/my_npy_dataset/x_train.npy")
    y_train = np.load("../dataset/mtat/my_npy_dataset/y_train.npy")
    x_valid = np.load("../dataset/mtat/my_npy_dataset/x_valid.npy")
    y_valid = np.load("../dataset/mtat/my_npy_dataset/y_valid.npy")
    x_test = np.load("../dataset/mtat/my_npy_dataset/x_test.npy")
    y_test = np.load("../dataset/mtat/my_npy_dataset/y_test.npy")
    x_train = x_train.reshape(-1, 59049, 1)
    x_valid = x_valid.reshape(-1, 59049, 1)
    x_test = x_test.reshape(-1, 59049, 1)

    return x_train, y_train, x_valid, y_valid, x_test, y_test


def lr_step_decay(epoch, lr):
    drop_rate = 0.2
    epochs_drop = 40.0
    initial_lr = 0.001
    print(initial_lr * math.pow(drop_rate, math.floor(epoch / epochs_drop)))

    return initial_lr * math.pow(drop_rate, math.floor(epoch / epochs_drop))


def train(initial_lr=0.01):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()
    train_data = TrainLoader(root="../dataset", split="train")
    valid_data = DataLoader(root="../dataset", split="valid")
    test_data = DataLoader(root="../dataset", split="test")
    model = music_sincnet()
    #optimizer = SGD(lr=initial_lr, momentum=args.momentum, decay=1e-6, nesterov=True)
    #model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[AUC()])
    csv_logger = CSVLogger("./training.csv", append=True)
    checkpoint_path = "ckeckpoint/cp.cpkt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )
    lr_scheduler = LearningRateScheduler(lr_step_decay, verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=4)
    
    for i in range(4):
        if i == 0:
            optimizer = Adam(learning_rate=1e-4)
            epoch = 60

        elif i==1:
            optimizer = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
            epoch = 20
        elif i==2:
            optimizer = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
            epoch=20

        else:
            optimizer = SGD(learning_rate=0.00001, momentum=0.9, nesterov=True)
            epoch=100
    
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=[AUC(name='auc')])
        model.fit(
            train_data,
            batch_size=16,
            epochs=epoch,
            callbacks=[checkpointer, csv_logger],
            validation_data = valid_data,
        )
        model.save("./model/rese_model"+str(i)+".h5")

    #y_prob = model.predict(x_test)
    #print("test auc :", metrics.roc_auc_score(y_test, y_prob))
    
        # test code
        print("evaluate result", model.evaluate(test_data))

    

if __name__ == "__main__":
    train()
