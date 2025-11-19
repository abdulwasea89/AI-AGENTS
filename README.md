# AI Agents Playground 🤖

A comprehensive collection of AI agent implementations and experiments using various frameworks including OpenAI Agents SDK, AutoGen, LangGraph, CrewAI, and more.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Agents SDK](#agents-sdk)
  - [AutoGen](#autogen)
  - [LangGraph](#langgraph)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This repository serves as a learning playground for exploring different AI agent frameworks and patterns. Each subdirectory contains examples and experiments with specific frameworks:

- **agents_sdk**: OpenAI Agents SDK examples (agent cloning, tool forcing)
- **auto_gen**: Microsoft AutoGen framework implementations
- **langgraph**: LangGraph supervisor patterns
- **crew_ai**: CrewAI multi-agent systems
- **agno**: Agno framework experiments
- **mastra**: Mastra framework examples
- **n8n**: N8N workflow automation integrations
- **rag_ai_agents**: RAG (Retrieval-Augmented Generation) implementations
- **graph_db_wit_ai_agents**: Graph database integrations
- **deployment_for_ai_agents**: Deployment configurations
- **docker_for_ai_agents**: Docker containerization
- **prod_ai_agents**: Production-ready agent patterns

## 📁 Project Structure

```
AI AGENTS/
├── agents_sdk/          # OpenAI Agents SDK examples
│   ├── main.py
│   ├── cloning_agent.py
│   ├── forcing_tool_use.py
│   └── pyproject.toml
├── auto_gen/            # AutoGen framework
│   ├── main.py
│   └── pyproject.toml
├── langgraph/           # LangGraph examples
│   └── supervsor.py
├── n8n/                 # N8N workflows
├── .venv/               # Virtual environment
└── README.md
```

## ✅ Prerequisites

- **Python**: 3.12 or higher
- **Package Manager**: `uv` (recommended) or `pip`
- **API Keys**: 
  - Google Gemini API key (or OpenAI API key)
  - Optional: Tavily, SerpAPI for web search capabilities

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd "AI AGENTS"
```

### 2. Set Up Virtual Environment

Using `uv` (recommended):
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

#### For Agents SDK:
```bash
cd agents_sdk
uv pip install -e .
# or with pip:
pip install -e .
```

Dependencies installed:
- `openai-agents>=0.5.1`
- `langchain-community>=0.4.1`
- `duckduckgo-search>=8.1.1`
- `ddgs>=9.9.1`

#### For AutoGen:
```bash
cd auto_gen
uv pip install -e .
# or with pip:
pip install -e .
```

Dependencies installed:
- `autogen[websurfer]>=0.10.1`
- `autogen-agentchat>=0.7.5`
- `autogen-ext[anthropic,google,groq,mistral,openai,together]>=0.7.5`
- `tavily-python>=0.7.13`
- `serpapi>=0.1.5`

#### For LangGraph:
```bash
pip install langgraph langchain-openai python-dotenv
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in each project directory:

**For agents_sdk/.env and auto_gen/.env:**
```env
GEMINI_API_KEY=your_gemini_api_key_here
# Or for OpenAI:
# OPENAI_API_KEY=your_openai_api_key_here

# Optional for web search:
TAVILY_API_KEY=your_tavily_key_here
SERPAPI_KEY=your_serpapi_key_here
```

**Getting API Keys:**
- **Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **OpenAI**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily**: Get from [Tavily](https://tavily.com/)
- **SerpAPI**: Get from [SerpAPI](https://serpapi.com/)

## 🎮 Usage

### Agents SDK

#### Basic Agent
```bash
cd agents_sdk
python main.py
```

This runs a simple assistant agent with Gemini 2.5 Flash.

#### Agent Cloning
```bash
python cloning_agent.py
```

Demonstrates how to clone agents with different instructions while sharing the same model configuration.

#### Forcing Tool Use
```bash
python forcing_tool_use.py
```

Shows how to force agents to use specific tools and integrate web search capabilities.

**Code Example:**
```python
import os
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

external_client = AsyncOpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

llm_model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model=llm_model
)

runner = Runner.run_sync(starting_agent=agent, input="Hello!")
print("AGENT RESPONSE:", runner.final_output)
```

### AutoGen

#### Basic AutoGen Agent
```bash
cd auto_gen
python main.py
```

Runs an AutoGen agent that can search for information and respond to queries.

**Code Example:**
```python
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

    result = await agent.run(task="Find information on AutoGen")
    print(result.messages)

if __name__ == "__main__":
    asyncio.run(main())
```

### LangGraph

#### Supervisor Pattern
```bash
cd langgraph
python supervsor.py
```

Demonstrates a supervisor agent pattern for coordinating multiple specialized agents.

## 📚 Examples

### Example 1: Web Search Agent

```python
from agents import Agent, function_tool, ModelSettings
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

@function_tool
def search_web(query: str) -> str:
    """Search the web for real-time information"""
    return search.run(query)

agent = Agent(
    name="SearchAgent",
    instructions="You are a helpful assistant with web search capabilities",
    model=llm_model,
    tools=[search_web],
    model_settings=ModelSettings(tool_choice="auto", temperature=0.1)
)

runner = Runner.run_sync(
    starting_agent=agent,
    input="What is the latest news about AI?"
)
print(runner.final_output)
```

### Example 2: Custom Tool Creation

```python
from agents import function_tool

@function_tool
def get_weather(city: str) -> str:
    """Get weather information for a city"""
    # In production, call a real weather API
    return f"The weather in {city} is sunny and 72°F"

agent = Agent(
    name="WeatherBot",
    instructions="Help users with weather information",
    model=llm_model,
    tools=[get_weather]
)
```

## 🔧 Troubleshooting

### Common Issues

**ImportError: cannot import name 'OpenAIChatCompletionsModel'**
- Solution: Use `OpenAIChatCompletionClient` instead (AutoGen framework)

**SyntaxError: 'await' outside function**
- Solution: Wrap async code in `async def main()` and use `asyncio.run(main())`

**API Key Not Found**
- Solution: Ensure `.env` file exists and `load_dotenv()` is called before accessing environment variables

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is for educational purposes. Individual frameworks may have their own licenses.

## 🔗 Resources

- [OpenAI Agents SDK](https://github.com/openai/openai-agents)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
- [CrewAI](https://www.crewai.com/)
- [Google Gemini](https://ai.google.dev/)

---

**Happy Agent Building! 🚀**
