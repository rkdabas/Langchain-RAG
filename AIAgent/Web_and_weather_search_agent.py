from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests
import os
from dotenv import load_dotenv

load_dotenv()


# Step1: Create a tool and LLM

# Search tool
search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("What happended to AirIndia flight 812?")
# print(results)

# Weather tool
@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'
  response = requests.get(url)
  return response.json()


# LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0)

# Now, we have LLM and search tool. So we can make our AI Agent

# Step2: Pull the ReAct prompt from langchain Hub
prompt = hub.pull("hwchase17/react")

# Step3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)

# Step4: Create the AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)

# Step5: Run the AgentExecutor
result = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(result)
print(result["output"])