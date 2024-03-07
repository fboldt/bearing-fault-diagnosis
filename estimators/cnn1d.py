from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', epochs=100):
        self.optimizer = optimizer
        self.epochs = epochs
        self.model = None
        self.convLayers = Sequential([
            layers.Conv1D(32, 64, activation='relu', name="conv1"),
            layers.MaxPooling1D(8),
            layers.Conv1D(32, 64, activation='relu', name="conv2"),
        ])

    def make_model(self, input_shape, num_classes):
        self.model = Sequential()
        self.model.add(layers.InputLayer(input_shape=input_shape))
        self.model.add(self.convLayers)
        self.model.add(layers.MaxPooling1D(8))
        self.model.add(layers.Conv1D(32, 64, activation='relu', name="conv3"))
        self.model.add(layers.GlobalAveragePooling1D(name='G_A_P_1D'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))


    def prefit(self, X, y):
        optimizer = self.optimizer
        epochs = self.epochs

        self.n_steps = X.shape[1]
        self.n_features = X.shape[2]

        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]

        self.make_model((self.n_steps, self.n_features), num_classes)
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=["categorical_accuracy"])
        self.model.fit(X, y_cat, epochs=epochs, verbose=False)
        self.convLayers.trainable = False
        print(self.model.summary())


    def fit(self, X, y=None):
        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        if self.model == None:
            self.n_steps = X.shape[1]
            self.n_features = X.shape[2]
            num_classes = y_cat.shape[1]
            self.make_model((self.n_steps, self.n_features), num_classes)
        optimizer = self.optimizer
        epochs = self.epochs

        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=["categorical_accuracy"])
        self.model.fit(X, y_cat, epochs=epochs, verbose=False)

    def predict(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        predictions = self.model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]
