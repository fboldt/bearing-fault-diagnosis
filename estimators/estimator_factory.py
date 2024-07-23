import importlib
import inspect
import os
from sklearn.base import BaseEstimator, ClassifierMixin

class EstimatorFactory:
    def __init__(self, estimators_dir=os.path.dirname(__file__)):
        self.estimators_dir = estimators_dir
        self.estimators = self._find_estimators()
        self.current_estimator = None

    def _find_estimators(self):
        estimators = {}
        for file in os.listdir(self.estimators_dir):
            if file.endswith('.py') and file != '__init__.py' and file != 'estimator_factory.py':
                module_name = file[:-3]
                estimators[module_name] = None
        return estimators

    def set_estimator(self, name):
        if name in self.estimators:
            if self.estimators[name] is None:
                module = importlib.import_module(f'estimators.{name}')
                for attr_name, attr_value in inspect.getmembers(module, inspect.isclass):
                    if issubclass(attr_value, (BaseEstimator, ClassifierMixin)) and attr_value.__module__ == module.__name__ and attr_value != BaseEstimator:
                        self.estimators[name] = attr_value
                        break
                if self.estimators[name] is None:
                    raise ValueError(f"No Estimator class found in module 'estimators.{name}'")
            self.current_estimator = self.estimators[name]
        else:
            raise ValueError(f'Estimator {name} not found')

    def get_estimator(self):
        if self.current_estimator is None:
            raise ValueError('No estimator has been set')
        return self.current_estimator()
