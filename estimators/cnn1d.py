from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CNN1D(BaseEstimator, ClassifierMixin):
    def __init__(self, optimizer='adam', epochs=100):
        self.optimizer = optimizer
        self.epochs = epochs

    def fit(self, X, y=None):
        optimizer = self.optimizer
        epochs = self.epochs

        # Define input shapes
        self.n_steps = X.shape[1]
        self.n_features = X.shape[2]

        self.labels, ids = np.unique(y, return_inverse=True)
        y_cat = to_categorical(ids)
        num_classes = y_cat.shape[1]

        self.model = Sequential()
        self.model.add(layers.InputLayer(input_shape=(self.n_steps, self.n_features)))
        self.model.add(layers.Conv1D(32, 64, activation='relu'))
        self.model.add(layers.MaxPooling1D(8))
        self.model.add(layers.Conv1D(32, 64, activation='relu'))
        self.model.add(layers.GlobalAveragePooling1D(name='G_A_P_1D'))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(num_classes))
        self.model.add(layers.Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=optimizer,
                           metrics=["categorical_accuracy"])
        self.model.fit(X, y_cat, epochs=epochs, verbose=False)

    def predict(self, X, y=None):
        X = X.reshape((X.shape[0], X.shape[1], self.n_features))
        predictions = self.model.predict(X)
        return self.labels[np.argmax(predictions, axis=1)]
