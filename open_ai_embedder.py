import concurrent.futures
from time import sleep

from openai import OpenAI


class OpenAiEmbedder:

    def __init__(
            self,
            api_key,
            embedding_model="text-embedding-3-small"
    ):
        self.embedding_model = embedding_model
        self.client = OpenAI(api_key=api_key)


    def embed(self, text):
        response = self.client.embeddings.create(
            input=text, model=self.embedding_model
        )
        return response.data[0].embedding


    def batch_embed(self, texts):
        def embed_text(text):
            return self.embed(text)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            sleep(0.2)
            embeddings = list(executor.map(embed_text, texts))

        return embeddings
