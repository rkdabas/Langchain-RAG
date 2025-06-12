from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

prompt = PromptTemplate(
    template='Write a short summary of the following text: {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader('ants.txt',encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

result = chain.invoke({'text':docs[0].page_content})

print(result)