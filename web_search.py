from typing import Sequence

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.messages import ChatMessage # AgentEvent ??
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.ui import Console
#from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient

import asyncio as asyncio
import os

#%pip install "autogen-agentchat==0.4.0.dev11" "autogen-ext[openai]==0.4.0.dev11" 

# Load the environment variables from the .env file
from dotenv import load_dotenv
load_dotenv()


# src: https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/tutorial/selector-group-chat.html#example-web-search-analysis

# ![agents](https://microsoft.github.io/autogen/dev/_images/selector-group-chat.svg)
# This system uses three specialized agents:
# Planning Agent: The strategic coordinator that breaks down complex tasks into manageable subtasks.
# Web Search Agent: An information retrieval specialist that interfaces with the search_web_tool.
# Data Analyst Agent: An agent specialist in performing calculations equipped with percentage_change_tool.

# Note: This example uses mock tools instead of real APIs for demonstration purposes
def search_web_tool(query: str) -> str:
    if "2006-2007" in query:
        return """Here are the total points scored by Miami Heat players in the 2006-2007 season:
        Udonis Haslem: 844 points
        Dwayne Wade: 1397 points
        James Posey: 550 points
        ...
        """
    elif "2007-2008" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2007-2008 is 214."
    elif "2008-2009" in query:
        return "The number of total rebounds for Dwayne Wade in the Miami Heat season 2008-2009 is 398."
    return "No data found."


def percentage_change_tool(start: float, end: float) -> float:
    return ((end - start) / start) * 100


async def main() -> None:
    model_client=AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    )

    planning_agent = AssistantAgent(
        "PlanningAgent",
        description="An agent for planning tasks, this agent should be the first to engage when given a new task.",
        model_client=model_client,
        system_message="""
        You are a planning agent.
        Your job is to break down complex tasks into smaller, manageable subtasks.
        Your team members are:
            Web search agent: Searches for information
            Data analyst: Performs calculations

        You only plan and delegate tasks - you do not execute them yourself.

        When assigning tasks, use this format:
        1. <agent> : <task>

        After all tasks are complete, summarize the findings and end with "TERMINATE".
        """,
    )

    web_search_agent = AssistantAgent(
        "WebSearchAgent",
        description="A web search agent.",
        tools=[search_web_tool],
        model_client=model_client,
        system_message="""
        You are a web search agent.
        Your only tool is search_tool - use it to find information.
        You make only one search call at a time.
        Once you have the results, you never do calculations based on them.
        """,
    )

    data_analyst_agent = AssistantAgent(
        "DataAnalystAgent",
        description="A data analyst agent. Useful for performing calculations.",
        model_client=model_client,
        tools=[percentage_change_tool],
        system_message="""
        You are a data analyst.
        Given the tasks you have been assigned, you should analyze the data and provide results using the tools provided.
        """,
    )


    text_mention_termination = TextMentionTermination("TERMINATE")
    max_messages_termination = MaxMessageTermination(max_messages=25)
    termination = text_mention_termination | max_messages_termination

    team = SelectorGroupChat(
        [planning_agent, web_search_agent, data_analyst_agent],
        model_client=model_client,
        termination_condition=termination,
    )

    task = "Who was the Miami Heat player with the highest points in the 2006-2007 season, and what was the percentage change in his total rebounds between the 2007-2008 and 2008-2009 seasons?"

    # Use asyncio.run(...) if you are running this in a script.
    await Console(team.run_stream(task=task))
    # async for stream in team.run_stream(task=task):
    #     if isinstance(stream, TaskResult):
    #         print(f"Stop reason: { stream.stop_reason }")
    #     else:
    #         print(stream)


asyncio.run(main())