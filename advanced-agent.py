import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from dataclasses import dataclass

load_dotenv()

@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_fahrenheit: float
    humidity: float

@tool('get_weather', description="Return weather", return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()

@tool('locate_user', description="Look up a user's city based on a context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'ABC123':
            return 'New York'
        case 'CDE456':
            return 'London'
        case _:
            return 'Unknown'

model = init_chat_model('gpt-4.1-mini', temperature=0.3)

checkpointer = InMemorySaver()

agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],
    system_prompt = 'You are a helpful weather assistent, who always cracks jokes and is humorous while remaining helpful',
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer
)

config = {''}

agent_response = agent.invoke({
    'messages': [
        {'role': 'user',
         'content': 'What is the weather like New York?'}
    ]
})

print(agent_response['messages'][-1].content)