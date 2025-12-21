from services.base_interface import SearchService

class LocalSearchService(SearchService):
    def search(self, query:str)->str:
        return f"<b>Local result</b><br/>Query was: {query}"