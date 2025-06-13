from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]
# initialize the embedding model
embedding_model = OpenAIEmbeddings()

# create a FAISS vector store in memory
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# enable MMP in the retriever
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5}
)

# query the retriever
query = "What is langchain?"

result = retriever.invoke(query)

print(result)

for i,doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"content:\n{doc.page_content}")