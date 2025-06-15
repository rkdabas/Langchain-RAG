from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

# create a tool
@tool
def multiply(a:int,b:int)->int:
    """given two numbers, multiply them"""
    return a*b

print(multiply.invoke({"a":10,"b":20}))


# initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

# bind the tools to the LLM i.e now LLM has access to the tool too.
llm_with_tools = llm.bind_tools([multiply])

print(llm_with_tools.invoke("what is 2*3?"))  # here the content in output is empty.
print(llm_with_tools.invoke("what is 2*3?").tool_calls)
 











