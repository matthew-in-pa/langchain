from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

@tool
def add(a: int, b: int) -> int:
    """Adds a and b"""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """multiplies a and b"""
    return a * b

tools = [add,multiply]

llm = init_chat_model(model="mistral:latest", model_provider="ollama", temperature=0)

llm_with_tools = llm.bind_tools(tools)

#query = "what is 3 * 12?"
#query = "What is 3 * 12 and 12 + 34 as well as 45 * 2"
query = "what is 4 plus 9, 32 times 4"

messages = [HumanMessage(query)]

ai_msg = llm_with_tools.invoke(messages)

for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_msg = selected_tool.invoke(tool_call)
    print(tool_msg)

