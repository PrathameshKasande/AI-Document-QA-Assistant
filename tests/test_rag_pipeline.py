from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.rag_pipeline import get_conversation_chain

def test_rag_pipeline():

    chunks = [
        "Deep Learning uses neural networks.",
        "NLP is used for language processing."
    ]

    embeddings = get_embedding_model()

    vector_store = create_vector_store(
        chunks,
        embeddings
    )

    conversation_chain = get_conversation_chain(
        vector_store
    )

    assert conversation_chain is not None