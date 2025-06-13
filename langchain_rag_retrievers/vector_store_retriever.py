from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# your source documents
documents = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

# initialize the embedding model
embedding_model = OpenAIEmbeddings()

# create chroma vector store in memory
vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="langchain_chroma_collection"
)

# convert vector store into a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# query the retriever
query = "What vector database is used for LLM-based search?"

result = retriever.invoke(query)

print(result)

for i,doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"content:\n{doc.page_content}")