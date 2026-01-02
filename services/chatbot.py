from services.base_interface import ChatService

class ChatServiceBase(ChatService):
    def __init__(self, ):
        print('chat initiated')

    def chat(self, query):
        return (f"No LLM Available: your query was {query}")
    
class ChatServiceLLM(ChatService):
    def __init__(self, ):
        print('chat initiated')

    def chat(self, query):
        print('querying')

        return (f"your query was {query}")