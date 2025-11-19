import asyncio
import os
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

async def main():
    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="You are a helpful assistant",
    )

    # Use asyncio.run(agent.run(...)) when running in a script.
    result = await agent.run(task="Find information on AutoGen")
    print(result.messages)

if __name__ == "__main__":
    asyncio.run(main())