from abc import ABC, abstractmethod

class SearchService(ABC):
    @abstractmethod
    def search(self, query: str) -> str:
        pass

class ChatService(ABC):
    @abstractmethod
    def chat(self, query:str) -> str:
        pass