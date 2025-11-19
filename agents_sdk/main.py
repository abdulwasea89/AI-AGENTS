import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
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

agent = Agent(name="Assistant", instructions="You are a helpful assistant",  model=llm_model)

runner = Runner.run_sync(starting_agent=agent, input="Hey")


print("AGENT RESPONSE: " , runner.final_output)