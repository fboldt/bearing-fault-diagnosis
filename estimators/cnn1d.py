from tensorflow.keras import layers, callbacks, saving
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import os.path
import shutil

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', epochs=1000, checkpoint="model.checkpoint"):
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = None
        self.checkpoint = checkpoint
        self.prefitckp = "prefit.checkpoint"
        self.validation_split = 0.1
        self.verbose = 2
        self.featLayers = Sequential(name="feat_layers")
        for i, (filters, kernel) in enumerate(zip([32, 32],[64, 64])):
            self.featLayers.add(layers.Conv1D(filters, kernel, 
                                              activation='relu', 
                                              name=f"conv{i+1}",
                                              ))
            self.featLayers.add(layers.MaxPooling1D(4))
            self.featLayers.add(layers.Dropout(0.2))

    
    def __del__(self):
        if os.path.isdir(self.checkpoint):
            if self.verbose:
                print("removing", self.checkpoint)
            shutil.rmtree(self.checkpoint)
        if os.path.isdir(self.prefitckp):
            if self.verbose:
                print("removing", self.prefitckp)
            shutil.rmtree(self.prefitckp)
    
    def callbacks_list(self, checkpoint=None):
        checkpoint = checkpoint if checkpoint else self.checkpoint
        monitor = "val_loss" if self.validation_split else "loss"
        return [
            callbacks.ModelCheckpoint(
                filepath=checkpoint,
                monitor=monitor,
                save_best_only=True,
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=100,
            )
        ]

    def make_model(self, input_shape):
        self.model = Sequential(name="backbone")
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.model.add(self.featLayers)
        self.model.add(layers.Conv1D(32, 32, activation='relu', name=f"conv_backbone"))
        self.model.add(layers.GlobalAveragePooling1D(name='gap1d'))
        self.model.add(layers.Dropout(0.5))
    
    def training(self, X, y, checkpoint):
        self.n_steps = X.shape[1]
        self.n_features = X.shape[2]

        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]

        self.make_model((self.n_steps, self.n_features))
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))

        self.model.compile(
            optimizer=self.optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )
        self.model.fit(X, y_cat, 
                       epochs=self.epochs, 
                       verbose=self.verbose,
                       callbacks=self.callbacks_list(checkpoint),
                       validation_split=self.validation_split)
        if os.path.isdir(checkpoint):
            if self.verbose:
                print("loading", checkpoint)
            model = saving.load_model(checkpoint)
            print(model.evaluate(X, y_cat))

    def prefit(self, X, y):
        self.training(X, y, self.prefitckp)
        self.featLayers.trainable = False
        print(self.model.summary())

    def fit(self, X, y=None):
        if os.path.isdir(self.prefitckp):
            print("loading", self.prefitckp)
            self.model = saving.load_model(self.prefitckp)
        if self.model == None:
            self.n_steps = X.shape[1]
            self.n_features = X.shape[2]
            self.make_model((self.n_steps, self.n_features))
        else:
            self.model.pop()
            self.model.pop() 
        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))
        self.training(X, y, self.checkpoint)

    def predict(self, X):
        if os.path.isdir(self.checkpoint):
            if self.verbose:
                print("loading", self.checkpoint)
            model = saving.load_model(self.checkpoint)
        else: 
            model = self.model
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        predictions = model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]