from langchain_community.vectorstores import FAISS


def create_vector_store(
    chunks,
    embeddings
):

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    return vector_store