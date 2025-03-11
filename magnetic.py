
import asyncio
import os
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console
from dotenv import load_dotenv
load_dotenv()


async def main() -> None:
    # model_client = OpenAIChatCompletionClient(model="gpt-4o")
    model_client=AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )            


    assistant = AssistantAgent(
        "Assistant",
        model_client=model_client,
    )
    team = MagenticOneGroupChat([assistant], model_client=model_client)
    await Console(team.run_stream(task="Provide a different proof for Fermat's Last Theorem"))


asyncio.run(main())


