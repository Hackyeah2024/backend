from llm_models import embeddings


def get_embeddings(text):
    embedding = embeddings.embed_query(text)
    return embedding
