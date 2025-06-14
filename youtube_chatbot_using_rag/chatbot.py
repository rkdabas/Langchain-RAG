from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv


load_dotenv()



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
retriever.invoke("What is deepmind")






# Step3: Augmentation
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
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

question = " is the topic of langchain discussed in this video? If yes then what was discussed?"
retrieved_docs = retriever.invoke(question)
print(retrieved_docs)

# concatenate the page_content from the 4 retrieved documents to get the final context
context = "\n\n".join(doc.page_content for doc in retrieved_docs)
print(context)

final_prompt = prompt.invoke({'context':context,'question':question})
print(final_prompt)







# Step3: Generation
answer = llm.invoke(final_prompt)
print(answer)












