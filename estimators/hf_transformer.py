from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import pipeline

class HFTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.model = pipeline("audio-classification")
    def fit(self, X, y):
        pass
    def predict(self, X):
        return self.model(X)

class Constructor():
    def __init__(self) -> None:
        pass
    def estimator(self):
        return HFTransformer()


# from datasets import load_dataset
# from datasets import Audio

# minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
# minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

# classifier = pipeline("audio-classification")

# example = minds[0]

# output = classifier(example["audio"]["array"])

# print(output)