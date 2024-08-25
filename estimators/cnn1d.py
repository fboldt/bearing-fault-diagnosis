from tensorflow.keras import layers, callbacks, optimizers
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import label_binarize
import numpy as np
import os

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=100, checkpoint="model.checkpoint.keras", verbose=2):
        self.optimizer = optimizers.Adam(learning_rate=0.001)
        self.epochs = epochs
        self.model = None
        self.checkpoint = checkpoint
        self.prefitckp = "prefit.checkpoint.keras"
        self.verbose = verbose
        self.validation_split = 0.2
    
    def __str__(self):
        if self.model:
               return f"{self.model.summary(expand_nested=True)}"
        return "CNN1D"
    
    # def __del__(self):
    #     self.remove_chepoint_file()
    #     self.remove_prefitckp_file()

    def remove_chepoint_file(self):
        if os.path.exists(self.checkpoint):
            if self.verbose:
                print("removing", self.checkpoint)
            os.remove(self.checkpoint)
        if os.path.exists(self.checkpoint+".labels"):
            os.remove(self.checkpoint+".labels")

    def remove_prefitckp_file(self):
        if os.path.exists(self.prefitckp):
            if self.verbose:
                print("removing", self.prefitckp)
            os.remove(self.prefitckp)
    
    def callbacks_list(self, checkpoint=None):
        checkpoint = checkpoint if checkpoint else self.checkpoint
        monitor = "val_accuracy" # "val_loss" # 
        return [
            callbacks.ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=self.epochs//10,
            )
        ]
    
    def make_feature_layers(self):
        self.featLayers = Sequential(name="feat_layers")
        filters = [32, 64, 128]
        kernels = [32 for _ in range(len(filters))]
        for i, (filter, kernel) in enumerate(zip(filters[:-1], kernels[:-1])):
            self.featLayers.add(layers.Conv1D(filter, kernel, 
                                              activation='relu', 
                                              name=f"conv_kernel{kernel}_{i+1}",
                                              ))
            self.featLayers.add(layers.AveragePooling1D(2, name=f"maxpool_{i+1}"))
        self.featLayers.add(layers.Conv1D(filters[-1], kernels[-1], 
                                          activation='relu', 
                                          name=f"conv_kernel{kernel}_last",
                                          ))
        self.featLayers.add(layers.GlobalMaxPooling1D(name='flat'))
        return self.featLayers

    def make_model(self, input_shape, num_classes):
        self.model = Sequential(name="backbone")
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.model.add(layers.BatchNormalization())
        self.model.add(self.make_feature_layers())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))
    
    def training(self, X, y, Xva=None, yva=None, checkpoint=None):
        self.n_steps = X.shape[1]
        self.n_features = X.shape[2]
        self.labels, ids = np.unique(y, return_inverse=True)
        with open(checkpoint+".labels", 'wb') as f:
            np.save(f, self.labels)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]
        if os.path.exists(self.prefitckp):
            print("loading", self.prefitckp)
            self.model = load_model(self.prefitckp)
            self.model.layers[0].trainable = False
        if self.model == None:
            self.make_model((self.n_steps, self.n_features), num_classes)
        else:
            self.model.pop()
            self.model.pop() 
            self.model.add(layers.Dense(num_classes))
            self.model.add(layers.Activation('softmax'))
        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
        if Xva is None or yva is None:
            Xtr, Xva, ytr, yva_cat = train_test_split(X, y_cat,
                                                      test_size=self.validation_split,
                                                      stratify=y)
        else:
            yva_cat = label_binarize(yva, classes=self.labels)
            Xtr, ytr = X, y_cat
        self.model.fit(Xtr, ytr, 
                       epochs=self.epochs, 
                       verbose=self.verbose,
                       callbacks=self.callbacks_list(checkpoint),
                       validation_data=(Xva, yva_cat))        
        if os.path.exists(checkpoint):
            if self.verbose:
                print("loading", checkpoint)
            self.model = load_model(checkpoint)

    def prefit(self, Xtr, ytr, Xva, yva):
        self.remove_prefitckp_file()
        self.training(Xtr, ytr, Xva, yva, self.prefitckp)
        self.model.layers[0].trainable = False

    def fit(self, Xtr, ytr, Xva=None, yva=None):
        self.remove_chepoint_file()
        self.training(Xtr, ytr, Xva, yva, self.checkpoint)

    def predict(self, X):
        if os.path.exists(self.checkpoint):
            if self.verbose:
                print("loading", self.checkpoint)
            model = load_model(self.checkpoint)
            with open(self.checkpoint+".labels", 'rb') as f:
                self.labels = np.load(f)
        else: 
            model = self.model
        predictions = model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]

