from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import requests    
from langchain_core.tools import InjectedToolArg
from typing import Annotated
import json

# create the tool to get the conversion factor
@tool
def get_conversion_factor(base_currency:str, target_currency:str)->float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency.
    """
    url = f"https://v6.exchangerate-api.com/v6/32edad6f76c69564a722c8ff/pair/{base_currency}/{target_currency}"
    response = requests.get(url,verify=False)
    return response.json()

# conversion_factor = get_conversion_factor.invoke({"base_currency":"EUR","target_currency":"INR"})
# print(conversion_factor)


# tool to multiply the conversion factor with the amount
@tool
def convert(base_currency_value:int, conversion_rate:Annotated[float,InjectedToolArg()])->float:
    """
    Given a currency conversion rate, this function calculates the target currency value from a given base currency value.
    """
    return base_currency_value * conversion_rate

# converted_amount = convert.invoke({"base_currency_value":100,"conversion_rate":conversion_factor["conversion_rate"]})
# print(converted_amount)


# initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

# bind the tools to the LLM
llm_with_tools = llm.bind_tools([get_conversion_factor,convert])

# create a query
query = HumanMessage(content="what is the conversion factor between EUR and INR and based on that can you convert 100 EUR to INR.")
messages = [query]

# get the AIMessage from LLM for the query
ai_message = llm_with_tools.invoke(messages)
# print(ai_message.tool_calls)

# tool calling
for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)

final_result = llm_with_tools.invoke(messages)
print(final_result.content)


    


