from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

def test_vector_store():

    chunks = [
        "Artificial Intelligence is a branch of computer science.",
        "Machine Learning is a subset of AI."
    ]

    embeddings = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    results = vector_store.similarity_search(
        "What is AI?"
    )

    assert len(results) > 0 