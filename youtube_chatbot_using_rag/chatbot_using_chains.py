from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv


load_dotenv()


def format_docs(retrieved_docs):
  context="\n\n".join(doc.page_content for doc in retrieved_docs)
  return context




# Step1: Indexing

# Part1: document loader i.e getting the entire transcript of the video
video_id = "uBL0siiliGo"
try:
  transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
  # flatten the transcript into plain text
  all_text_parts=[]
  for chunk in transcript_list:
    text=chunk["text"]
    all_text_parts.append(text)
  transcript = " ".join(all_text_parts)
  print(transcript_list)
  print(transcript)
except TranscriptsDisabled:
  print("no caption available for this video")

# Part2: Text splitter i.e splitting the transcript into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.create_documents([transcript])
print(len(chunks))
print(chunks[0])

# Part3: Embedding the chunks into vectors
embeddings = GoogleGenerativeAIEmbeddings(model_name="text-embedding-004")

# Part4: store the embeddings in a vector store with the help of FAISS. 
vector_store = FAISS.from_documents(chunks, embeddings)
vector_store.index_to_docstore_id
# vector_store.get_by_ids(['insert the id of the chunk'])





# Step2: Retrieval  (i/p is a query and o/p is a list of most similar chunks/documents)
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever)




# Step3: Create the parallel chain.
parallel_chain = RunnableParallel({
  'question':RunnablePassthrough(),
  'context': retriever | RunnableLambda(format_docs)  # here retriever is doing semantic search and giving list of documents and it is passed in another Runnable to format the docs
})

print(parallel_chain.invoke("What is deepmind"))




# Step4: Augmentation
prompt = PromptTemplate(
  template="""
  You are helpful assistant.
  Answer only from the provided transcript context.
  If the context is insufficient, just say you don't know.

  {context}
  Question: {question}
  """,
  input_variables=["context", "question"]
)

# Step5: Initialize the LLM.
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Step6: Initialize the parser.
parser = StrOutputParser()

# Step7: Create the main chain.
main_chain = parallel_chain | prompt | llm | parser

# Step8: Invoke the main chain.
result = main_chain.invoke("can you summarize this video")
print(result)

