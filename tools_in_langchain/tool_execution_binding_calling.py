from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests

# create a tool to multiply two numbers
@tool
def multiply(a:int,b:int)->int:
    """given two numbers, multiply them"""
    return a*b
# print(multiply.invoke({"a":10,"b":20}))


# initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

# bind the tools to the LLM i.e now LLM has access to the tool too.
llm_with_tools = llm.bind_tools([multiply])

# create a query(HumanMessage) and append it to the messages
query = HumanMessage(content="can you multiple 10 and 8")
messages = [query]

# get the AIMessage from LLM for the query
result = llm_with_tools.invoke(messages)
# print(llm_with_tools.invoke("what is 2*3?").tool_calls)

# append the AIMessage to the messages
messages.append(result)  # now, messages has AIMessage and HumanMessage.


# Tool execution i.e execute the multiply tool
tool_call_result = multiply.invoke(result.tool_calls[0])
print(tool_call_result)  # It returns (ToolMessage) object.


# now append the ToolMessage to the messages
messages.append(tool_call_result) # now messages has AIMessage, HumanMessage and ToolMessage.
print(messages)

# now pass the messages to the LLM to get the final result.
# LLM has access to the tool, all the messages.
final_result = llm_with_tools.invoke(messages)
print(final_result)
print(final_result.content)











