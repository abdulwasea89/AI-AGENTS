import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, ModelSettings
from langchain_community.tools import DuckDuckGoSearchRun
from dotenv import load_dotenv, find_dotenv

load_dotenv()

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

llm_model: OpenAIChatCompletionsModel = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)   

@function_tool
def get_weather(city: str) -> str:
    return f"The weater of this {city} is cold"

agent = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant",  
    model=llm_model,
    tools=[get_weather],
    model_settings=ModelSettings(tool_choice="required", temperature=0.1)
)

# runner = Runner.run_sync(starting_agent=agent, input="hey?")


search = DuckDuckGoSearchRun()

@function_tool
def search_web(query: str) -> str:
    """
    Search the web for real-time information
    Returns a text summary of results
    """
    results = search.run(query)
    return results


agent_web = Agent(
    name="Assistant", 
    instructions="You are a helpful assistant",  
    model=llm_model,
    tools=[search_web],
    model_settings=ModelSettings(tool_choice="auto", temperature=0.1)
)


runner_web = Runner.run_sync(starting_agent=agent_web, input="What is AI Mood in google search and the best model of the world write now")

print("AGENT RESPONSE: " , runner_web.final_output)