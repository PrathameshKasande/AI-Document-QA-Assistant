from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_text(text)

    # Remove empty chunks
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    return chunks