from tensorflow.keras import layers, callbacks, saving, optimizers, Model, Input
from tensorflow.keras.models import Sequential
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
        monitor = "val_loss" # "val_accuracy" # 
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
    
    def make_feature_layers(self, x):
        filters = [2**i for i in range(5,7)]
        kernel_size = 180
        kernels = [kernel_size for _ in range(len(filters))]
        sources = {
            "gear": (0, 6, None),
            "leftaxl": (6, 9, None),
            "motor": (9, 18, None),
        }
        convs = []
        for source in sources.keys():
            start, end, f = sources[source]
            f = x
            for (filter, kernel) in zip(filters, kernels):
                f = layers.Conv1D(filter, kernel, strides=int(kernel_size**0.5)+1, activation='relu')(f[:,:,start:end])
                f = layers.SpatialDropout1D(0.25)(f)
            f = layers.GlobalAveragePooling1D()(f)
            convs.append(f)
        x = layers.concatenate(convs, axis=-1)
        return x


    def make_model(self, input_shape, num_classes):
        inputs = Input(shape=input_shape)
        x = self.make_feature_layers(inputs)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(num_classes)(x)
        outputs = layers.Activation('softmax')(x)
        self.model = Model(inputs, outputs)
    
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
            self.model = saving.load_model(self.prefitckp)
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
            self.model = saving.load_model(checkpoint)

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
            model = saving.load_model(self.checkpoint)
            with open(self.checkpoint+".labels", 'rb') as f:
                self.labels = np.load(f)
        else: 
            model = self.model
        predictions = model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]

class Contructor():
    def __init__(self, epochs=100, checkpoint="model.checkpoint.keras", verbose=2):
        self.epochs=epochs
        self.checkpoint=checkpoint
        self.verbose=verbose
    def estimator(self):
        return CNN1D(epochs=self.epochs, checkpoint=self.checkpoint, verbose=self.verbose)
