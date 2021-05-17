import time
import datetime
import tensorflow.keras.backend as K
import os
from tqdm import tqdm
from model import resemul, HarmonicCNN
import tensorflow as tf

# fix random seed
SEED = 42
tf.random.set_seed(SEED)


class Solver(object):
    def __init__(self, config):
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        self.data_path = config.data_path
        self.model_save_path = config.model_save_path
        self.data_path = config.data_path
        self.block = config.block
        self.test_template = "Test Loss : {}, Test AUC : {:.4f}, Test PR-AUC : {:.4f}"
        self.train_template = "Epoch : {}, Loss : {:.4f}, AUC : {:.4f}, PR-AUC : {:.4f}"
        self.encoder_type = config.encoder_type
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        self.train_ds, self.valid_ds, self.test_ds = self.get_data(self.data_path, self.encoder_type)
        self.get_model()
        self.get_optimizer()
        self.get_loss()
        self.get_metric()
        self.start_time = time.time()


    def get_data(self, root="../../tf2-harmonic-cnn/dataset", encoder="HC"):
        if encoder == "HC":
            from data_loader.train_loader import TrainLoader
            from data_loader.mtat_loader import DataLoader
        else:
            from data_loader.rese_loader import TrainLoader
            from data_loader.rese_mtat_loader import DataLoader

        train_data = TrainLoader(root=root, split="train")
        valid_data = DataLoader(root=root, split="valid")
        test_data = DataLoader(root=root, split="test")
        return train_data, valid_data, test_data


    def get_model(self):
        if self.encoder_type == "HC":
            self.model = HarmonicCNN()
        elif self.encoder_type == "SC":
            self.model = resemul(block_type=self.block)
        elif self.encoder_type == "MS":
            self.model = resemul(ms=True, block_type="rese")
        

    def get_optimizer(self):
        self.adam = tf.keras.optimizers.Adam(learning_rate=1e-4)
        self.sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)


    def get_loss(self):
        self.c_loss = tf.keras.losses.BinaryCrossentropy()


    def get_metric(self):
        self.train_loss = tf.keras.metrics.Mean()
        self.train_auc = tf.keras.metrics.AUC()
        self.train_pr = tf.keras.metrics.AUC(curve='PR')
        self.test_loss = tf.keras.metrics.Mean()
        self.test_auc = tf.keras.metrics.AUC()
        self.test_pr = tf.keras.metrics.AUC(curve='PR')


    @tf.function
    def adam_step(self, wave, labels):
        with tf.GradientTape() as tape:

            predictions = self.model(wave, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.model.trainable_variables
        gradients = tape.gradient(loss, train_variable)
        
        self.adam.apply_gradients(zip(gradients, train_variable))

        self.train_loss(loss)
        self.train_auc(labels, predictions)
        self.train_pr(labels, predictions)


    @tf.function
    def sgd_step(self, wave, labels):
        with tf.GradientTape() as tape:

            predictions = self.model(wave, training=True)
            loss = self.c_loss(labels, predictions)

        train_variable = self.model.trainable_variables
        gradients = tape.gradient(loss, train_variable)

        self.sgd.apply_gradients(zip(gradients, train_variable))

        self.train_loss(loss)
        self.train_auc(labels, predictions)
        self.train_pr(labels, predictions)

    
    @tf.function
    def test_step(self, wave, labels):
        predictions = self.model(wave, training=False)
        loss = self.c_loss(labels, predictions)
        self.test_loss(loss)
        self.test_auc(labels, predictions)
        self.test_pr(labels, predictions)


    def train(self, epochs, optimizer):
        for epoch in range(epochs):
            for wave, labels in tqdm(self.train_ds):
                if optimizer == "adam":
                    self.adam_step(wave, labels)
                elif optimizer == "sgd":
                    self.sgd_step(wave, labels)

            stage2_log = self.train_template.format(
                epoch + 1, self.train_loss.result(), self.train_auc.result(), self.train_pr.result()
            )
            print(stage2_log)

        if (epoch % 19 == 0 and epoch!= 0):
            for valid_wave, valid_labels in tqdm(self.valid_ds):
                self.test_step(valid_wave, valid_labels)

            valid_log = self.test_template.format(self.test_loss.result(), self.test_auc.result(), self.test_pr.result())
            print(valid_log)

    

    def run(self):
        print("\n\n@@@@@@@@@@@@@@@@@@@Start training@@@@@@@@@@@@@@@@@@\n")
        for i in range(4):
            if i == 0:
                epochs = 60
                self.train(epochs=epochs, optimizer="adam")
            elif i == 1:
                epochs = 20
                self.train(epochs=epochs, optimizer="sgd")
            elif i ==2:
                epochs = 20
                new_lr = 0.0001
                self.sgd.lr.assign(new_lr)
                self.train(epochs=epochs, optimizer="sgd")
            else:
                epochs= 100
                new_lr = 0.00001
                self.sgd.lr.assign(new_lr)
                self.train(epochs=epochs, optimizer="sgd")

        for wave, labels in tqdm(self.test_ds):
            self.test_step(wave, labels)

        print("Time taken : ", time.time() - self.start_time)

        test_result = self.test_template.format(self.test_loss.result(), self.test_auc.result(), self.test_pr.result())
        print(test_result)


