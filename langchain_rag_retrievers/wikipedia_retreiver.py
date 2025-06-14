from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

query = "What is the capital of France?"

docs = retriever.invoke(query)

print(docs)

for i,doc in enumerate(docs):
    print(f"\n-- Result {i+1} --")
    print(f"content:\n{doc.page_content}")