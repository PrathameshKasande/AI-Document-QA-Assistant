from src.embeddings import get_embedding_model

def test_embedding_model():

    embeddings = get_embedding_model()

    vector = embeddings.embed_query(
        "What is artificial intelligence?"
    )

    assert isinstance(vector, list)

    assert len(vector) > 0