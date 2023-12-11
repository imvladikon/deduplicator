from abc import abstractmethod, ABC


class BaseNormalizer(ABC):

    def __call__(self, value: str) -> str:
        return self._normalize(value)

    @abstractmethod
    def _normalize(self, value: str) -> str:
        pass
