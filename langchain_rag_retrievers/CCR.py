from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_core.documents import Document


docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

# initialize the embedding model
embedding_model = OpenAIEmbeddings()

# create the vector store
vector_store = FAISS.from_documents(
    documents=docs,
    embedding=embedding_model
)

# create the base retriever for CCR
base_retriver = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# set up the compressor using an LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# create the CCR retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriver
)

# query the retriever
query = "What is the photosynthesis?"

result = compression_retriever.invoke(query)

print(result)

for i,doc in enumerate(result):
    print(f"\n-- Result {i+1} --")
    print(f"content:\n{doc.page_content}")
