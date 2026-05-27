from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

from src.llm_model import load_llm


custom_prompt = PromptTemplate(
    template="""
You are an intelligent AI assistant.

Use the provided context to answer the user's question in detail.

If the answer is not available in the context,
say:
"I could not find the answer in the uploaded PDFs."

Context:
{context}

Question:
{question}

Detailed Answer:
""",
    input_variables=["context", "question"]
)


def get_conversation_chain(vector_store):

    llm = load_llm()

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={
            "prompt": custom_prompt
        },
        return_source_documents=True
    )

    return conversation_chain