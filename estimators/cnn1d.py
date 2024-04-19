from tensorflow.keras import layers, callbacks, saving, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import label_binarize
import numpy as np
import os.path
import shutil

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', epochs=1000, checkpoint="model.checkpoint", verbose=2):
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = None
        self.checkpoint = checkpoint
        self.prefitckp = "prefit.checkpoint"
        self.validation_split = 0.2
        self.verbose = verbose
    
    def __str__(self):
        if self.model:
            return f"{self.model.summary(expand_nested=True)}"
        return "CNN1D"
    
    def __del__(self):
        self.remove_chepoint_file()
        self.remove_prefitckp_file()

    def remove_chepoint_file(self):
        if os.path.isdir(self.checkpoint):
            if self.verbose:
                print("removing", self.checkpoint)
            shutil.rmtree(self.checkpoint)

    def remove_prefitckp_file(self):
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
    
    def make_feature_layers(self):
        self.featLayers = Sequential(name="feat_layers")
        for i, (filters, kernel) in enumerate(zip([32, 32],[64, 64])):
            self.featLayers.add(layers.Conv1D(filters, kernel, 
                                              activation='relu', 
                                              name=f"conv_kernel{kernel}_{i+1}",
                                              kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                              bias_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),
                                              ))
            self.featLayers.add(layers.MaxPooling1D(4, name=f"maxpool_{i+1}"))
        self.featLayers.add(layers.GlobalMaxPooling1D(name='flat'))
        return self.featLayers

    def make_model(self, input_shape, num_classes):
        self.model = Sequential(name="backbone")
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.model.add(self.make_feature_layers())
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))
    
    def training(self, X, y, Xva=None, yva=None, checkpoint=None):
        self.n_steps = X.shape[1]
        self.n_features = X.shape[2]

        self.make_model((self.n_steps, self.n_features))
        self.labels, ids = np.unique(y, return_inverse=True)

        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))

        if os.path.isdir(self.prefitckp):
            print("loading", self.prefitckp)
            self.model = saving.load_model(self.prefitckp)
            self.model.layers[0].trainable = False
            # print(self.model.summary(expand_nested=True))

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
        
        if os.path.isdir(checkpoint):
            if self.verbose:
                print("loading", checkpoint)
            self.model = saving.load_model(checkpoint)
            # print("(Xtr, ytr)", model.evaluate(Xtr, ytr))
            # print("(Xva, yva)", model.evaluate(Xva, yva_cat))
            # print("(X, y)", model.evaluate(X, y_cat))

    def prefit(self, Xtr, ytr, Xva=None, yva=None):
        self.remove_prefitckp_file()
        self.training(Xtr, ytr, Xva, yva, self.prefitckp)
        self.model.layers[0].trainable = False

    def fit(self, Xtr, ytr=None, Xva=None, yva=None):
        self.training(Xtr, ytr, Xva, yva, self.checkpoint)

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
    