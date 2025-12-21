from abc import ABC, abstractmethod

class SearchService(ABC):
    @abstractmethod
    def search(self, querry: str) -> str:
        pass