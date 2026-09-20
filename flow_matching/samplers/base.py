from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample(self, model, **kwargs):
        ...
