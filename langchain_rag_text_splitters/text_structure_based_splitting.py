from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("metro_information.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
) 

chunks = splitter.split_documents(docs)

print(len(chunks))
print(chunks)