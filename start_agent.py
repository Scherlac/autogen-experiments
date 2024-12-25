from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import asyncio as asyncio
import os

#%pip install "autogen-agentchat==0.4.0.dev11" "autogen-ext[openai]==0.4.0.dev11" 

# Load the environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()

# Define a tool
async def get_weather(city: str) -> str:
    return f"The weather in {city} is 73 degrees and Sunny."


async  def main() -> None:
    # Define an agent
    weather_agent = AssistantAgent(
        name="weather_agent",
        model_client=AzureOpenAIChatCompletionClient(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        ),
        # model_client=OpenAIChatCompletionClient(
        #     model="gpt-4o-2024-08-06",
        #     api_key="YOUR_API_KEY",

        # ),
        tools=[get_weather],
    )

    # Define a team with a single agent and maximum auto-gen turns of 1.
    agent_team = RoundRobinGroupChat([weather_agent], max_turns=1)

    while True:
        # Get user input from the console.
        user_input = input("Enter a message (type 'exit' to leave): ")
        if user_input.strip().lower() == "exit":
            break
        # Run the team and stream messages to the console.
        # Adding observability 
        # src: https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/teams.html#observability
        async for stream in agent_team.run_stream(task=user_input):
            if isinstance(stream, TaskResult):
                print(f"Stop reason: { stream.stop_reason }")
            else:
                print(stream)
        # await Console(stream)


# NOTE: if running this inside a Python script you'll need to use asyncio.run(main()).
asyncio.run( main() )
