from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.lazy_load()

print(len(docs))

# now this loop will not load the documents into memory and print output
# very fastly
for doc in docs:
    print(doc.metadata)
