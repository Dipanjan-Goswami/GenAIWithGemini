class LoadProperties:

    def __init__(self):

        import json
        # reading the data from the file
        with open('config.json') as f:
            data = f.read()

        js = json.loads(data)

        self.model_name = js["model_name"]
        self.embedding_model = js["embed_model"]
        self.google_api_key = js["google_ai_api_key"]
        self.langchain_key = js["langchain_key"]
        self.langchain_endpoint = js["langchain_endpoint"]

    def getModelName(self):
        return self.model_name

    def getLangChainKey(self):
        return self.langchain_key

    def getlangChainEndpoint(self):
        return self.langchain_endpoint

    def getAPIkey(self):
        return self.google_api_key

    def getEmbeddingModel(self):
        return self.embedding_model