# Method 1: using @tool decorator
from langchain_core.tools import tool
@tool
def add(a:int, b:int)->int:
    """add two numbers"""
    return a+b

result = add.invoke({"a":10,"b":20})
print(result)
print(add.name)
print(add.description)
print(add.args)

# this is what LLM receives as input
print(add.args_schema.model_json_schema())







# Method 2: using StructuredTool and Pydantic
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="first number to multiple")
    b: int = Field(required=True, description="second number to multiple")

def multiply(a:int, b:int)->int:
    return a*b

multiply_tool = StructuredTool.from_function(
    func=multiply, 
    name="multiply", 
    description="multiply two numbers", 
    args_schema=MultiplyInput
)
print(multiply_tool.invoke({"a":10,"b":20}))
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema.model_json_schema())








# Method 3: using BaseTool
from langchain.tools import BaseTool
from typing import Type

class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="first number to multiple")
    b: int = Field(required=True, description="second number to multiple")

class multiplyTool(BaseTool):
    name:str = "multiply"
    description:str="multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self,a:int,b:int)->int:
        return a*b
    
multiply_tool = multiplyTool()
print(multiply_tool.invoke({"a":10,"b":20}))
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args_schema.model_json_schema())
