import asyncio
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.teams.magentic_one import MagenticOne
from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.messages import FunctionCall
from autogen_core import CancellationToken
from autogen_core.models import (
    UserMessage
)
from autogen_core import Component
from autogen_core.models import ChatCompletionClient

# from autogen_ext.agents.web_surfer import MultimodalWebSurfer


from autogen_core.tools import FunctionTool
from autogen_agentchat.ui import Console
import os
import json
import numpy as np
from pydantic import BaseModel



def google_search(query: str, num_results: int = 2, max_chars: int = 500) -> list:
    """
    Search Google for information and return the top results with a snippet and body content.

    Args:
        query: The search query.
        num_results: The number of results to return.
        max_chars: The maximum number of characters to return from the page content.

    Returns:
        list: A list of dictionaries containing the title, link, snippet, and body content of the search results.

    src: https://microsoft.github.io/autogen/dev/user-guide/agentchat-user-guide/examples/company-research.html#defining-tools
    """
    import os
    import time

    import requests
    from bs4 import BeautifulSoup

    api_key = os.getenv("GOOGLE_API_KEY")
    # SRC: https://programmablesearchengine.google.com/controlpanel/all
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        raise ValueError("API key or Search Engine ID not found in environment variables")

    url = "https://customsearch.googleapis.com/customsearch/v1"
    params = {"key": str(api_key), "cx": str(search_engine_id), "q": str(query), "num": str(num_results)}

    response = requests.get(url, params=params)

    if response.status_code // 100 != 2:
        print(response.json(), flush=True)
        raise Exception(f"Error in API request: {response.status_code}")

    results = response.json().get("items", [])

    def get_page_content(url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < max_chars:
                return text
            
            # poor man's similarity search:

            sentences = text.split(".") # split by sentence

            query_words = query.lower().split(" ")
            query_words = [word for word in query_words if len(word) > 3]
            minimum_matching_words = np.minimum(3, len(query_words))

            content = ""
            previous_sentence = ""
            for sentence in sentences:
                # Check if at least 3 query words are in the sentence
                sentence_lower = sentence.lower()
                matching_words = [word for word in query_words if word in sentence_lower]
                if len(matching_words) < minimum_matching_words:
                    previous_sentence = sentence + ". "
                    continue
                
                content += previous_sentence + sentence + ". "
                previous_sentence = ""

                if len(content) > max_chars:
                    break

            return content

        except Exception as e:
            print(f"Error fetching {url}: {str(e)}", flush=True)
            return ""

    enriched_results = []
    # num_results = 2
    for item in results[:num_results]:
        body = get_page_content(item["link"])
        enriched_results.append(
            {"title": item["title"], "link": item["link"], "snippet": item["snippet"], "body": body}
        )
        time.sleep(1)  # Be respectful to the servers

    return enriched_results    

google_search_tool = FunctionTool(google_search, description="Search Google for information.")

class SearchAgentConfig(BaseModel):
    name: str
    model_client: AzureOpenAIChatCompletionClient
    function_tool: FunctionTool


class SearchAgent(BaseChatAgent, Component[SearchAgentConfig]):
    def __init__(self, name: str, model_client: AzureOpenAIChatCompletionClient, function_tool: FunctionTool):
        super().__init__(name=name, description="Search Agent")
        self._model_client = model_client
        self._function_tool = function_tool

    async def on_messages(self, messages, cancellation_token):
        last_message = messages[-1]
        assert isinstance(last_message, UserMessage)

        task_content = last_message.content  # the last message from the sender is the task

        create_result = await self._model_client.create(
            messages=messages,
            tools=[self._function_tool],
            cancellation_token=cancellation_token,
        )

        response = create_result.content

        if isinstance(response, str):
            # Answer directly.
            return False, response

        elif isinstance(response, list) and all(isinstance(item, FunctionCall) for item in response):
            function_calls = response
            for function_call in function_calls:
                tool_name = function_call.name

                try:
                    arguments: dict = json.loads(function_call.arguments)
                except json.JSONDecodeError as e:
                    error_str = f"File surfer encountered an error decoding JSON arguments: {e}"
                    return False, error_str

                if tool_name == "google_search":
                    query = arguments["query"]
                    num_results = arguments.get("num_results", 2)
                    max_chars = arguments.get("max_chars", 500)
                    results = google_search(query, num_results=num_results, max_chars=max_chars)
                    return False, results

        return False, "No response from the search agent."

    async def on_reset(self, cancellation_token):
        pass

    # def _to_config(self) -> SearchAgentConfig:
    #     return SearchAgentConfig(
    #         name=self.name,
    #         model_client=self._model_client.dump_component(),
    #         function_tool=self._function_tool.dump_component(),
    #     )

async def example_usage():
    client=AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )            

    executor = DockerCommandLineCodeExecutor(
        image="pscrdevops210e54.azurecr.io/scl2bp/nvidia-2204:dev-latest",
        container_name="dev-scl2bp-agent",
        work_dir="/workspace/agent-workspace",
        stop_container=False,
        auto_remove=False,
    )


    m1 = MagenticOne(client=client,
            code_executor=executor,
            hil_mode=True,
            custom_agents=[
                SearchAgent("SearchAgent", model_client=client, function_tool=google_search_tool),
                # MultimodalWebSurfer("WebSurfer", model_client=client)

                ]
                     )
    task = "Write a Python script to fetch data from an API."
    result = await Console(m1.run_stream(task=task))
    print(result)


if __name__ == "__main__":
    asyncio.run(example_usage())