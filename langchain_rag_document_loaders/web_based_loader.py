from langchain_community.document_loaders import WebBaseLoader

url = "https://www.google.com"
# here we can pass multiple urls as a list too.
loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))