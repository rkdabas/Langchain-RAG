from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("pdf_file_for_pdf_loader.pdf")

docs = loader.load()

print(len(docs))