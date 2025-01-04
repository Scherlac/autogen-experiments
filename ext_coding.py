import os
import sys
import re
import asyncio

from dataclasses import dataclass
from typing import List, Any
import numpy as np

from autogen_core import (
    AgentId,
    DefaultInterventionHandler,
    DefaultTopicId, 
    DropMessage,
    FunctionCall,
    MessageContext, 
    RoutedAgent,
    SingleThreadedAgentRuntime,
    default_subscription, 
    message_handler
)
from autogen_core.code_executor import CodeBlock, CodeExecutor
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import tempfile

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.tools import FunctionTool
from autogen_core.tool_agent import ToolAgent, ToolException, tool_agent_caller_loop


# load .env file
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Message:
    content: str

@dataclass
class UserAssingment:
    content: str

@dataclass
class CommandMessage:
    command: str

@dataclass
class CommandResponse:
    output: str

@dataclass
class Search:
    query: str

@dataclass
class SearchResults:
    results: str

@dataclass
class Task:
    content: str

@dataclass
class TaskResults:
    results: str


@default_subscription
class PlanerAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        search_agent_type: str,
        assistant_agent_type: str

                 ) -> None:
        super().__init__(
            description="A planner agent that breaks down complex tasks into smaller tasks.",
            )
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                You are a planning agent.
                The job is to break down complex tasks into smaller tasks.

                Your team members are:
                    Search agent (id: {search_agent_type}): Searches for information.
                    Assisten agent (id: {assistant_agent_type}): Creates and executes scripts.

                Manage **plan** to accomplish the task as follows:
                ```python
                # filename: plan.py
                plan = [
                    {
                        "step": <step number>,
                        "description": "<step description>"
                    },
                    ...
                    ]
                ```

                Manage the list of **required information** and the **results** as follows:
                ```python
                # filename: requirements.py
                requirements = [
                    {
                        "id": <id>,
                        "description": "<requirement description>"
                        "target": "<target agent id>"
                        "result": "<result>"
                    },
                    ...
                    ]
                ```

                Manage the **tasks** and **prioritize** them as follows:
                ```python
                # filename: tasks.py
                tasks = [
                    {
                        "id": <task id>,
                        "agent_id": "<agent id>",
                        "description": "<task description>",
                        "priority": <task priority>
                    },
                    ...
                    ]
                ```

                Assign the next tasks based on the plan and priority, use this format:
                'TASK: <agent_id>, <task_id>, <task_description>'

                
                After all tasks are complete, summarize the findings and end with "TERMINATE".
                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.

            )
        ]

    @message_handler
    async def handle_message(self, message: UserAssingment, ctx: MessageContext) -> None:
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nPlanner:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="planner"))
        await self.publish_message(Task(content=result.content), DefaultTopicId())

    @message_handler
    async def handle_taskresult(self, message: TaskResults, ctx: MessageContext) -> None:
        self._chat_history.append(AssistantMessage(content=message.results, source="assistant"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nPlanner:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="planner"))
        await self.publish_message(Task(content=result.content), DefaultTopicId())



@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, 
            model_client: ChatCompletionClient,
            tool_agent_type: str
            ) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                Output markdown shell command to read script files or to verify the existence of a file.
                Output markdown script to complete the task.
                Output SEARCH: 'query text' if anditional information is required.
                Use existing script file as a starting point if requested.
                Use the current directory for file operations.
                The first line of the code block is as follows: '# filename: <filename>'.
                Work sequentially step by step to solve the task.
                Always provide only a single output for each step.
                Always save figures to file. Do not use show() or display() commands.
                Output SUCCESS: 'output file name', 'description and usage' if the task is completed.
                Output FAILURE: 'reason for failure' if the task is not completed.
                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.

            )
        ]
        self._tool_agent_type = tool_agent_type
        self._tool_agent = AgentId(tool_agent_type, f"Tool Agent - {self.id.key}")

    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> None:
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
        await self.send_message(CommandMessage(command=result.content), self._tool_agent)

    @message_handler
    async def handle_commandresponse(self, message: CommandResponse, ctx: MessageContext) -> None:
        self._chat_history.append(AssistantMessage(content=message.output, source="executor"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}", flush=True)
        if "SUCCESS" in result.content or "FAILURE" in result.content:
            await self.publish_message(TaskResults(results=result.content), DefaultTopicId())

        elif "SEARCH" in result.content:
            await self.publish_message(Search(query=result.content), DefaultTopicId())

        else:
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))
            await self.send_message(CommandMessage(command=result.content), self._tool_agent)


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor

    @message_handler
    async def handle_commandmessaga(self, message: CommandMessage, ctx: MessageContext) -> None:
        code_blocks = extract_markdown_code_blocks(message.command)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            print(f"\n{'-'*80}\nExecutor:\n{result.output}", flush=True)
            await self.publish_message(CommandResponse(output=result.output), DefaultTopicId())




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

@default_subscription
class SearchAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        tool_agent_type: str,
        # tools: List[Tool] = [google_search_tool]
                 ) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                You are a web search agent.
                Your only tool is search_tool - use it to find information.
                You make only one search call at a time.
                Once you have the results, you never do calculations based on them.
                """,
            )
        ]
        self._tool_agent_id = tool_agent_type
        # self._tools = tools

    @message_handler
    async def handle_message(self, message: Search, ctx: MessageContext) -> None:
        session: List[LLMMessage] = [UserMessage(content=message.query, source="user")]
        output_messages = await tool_agent_caller_loop(
            self,
            tool_agent_type=self._tool_agent_id,
            model_client=self._model_client,
            input_messages=session,
            tool_schema=self._tool_schema,
            cancellation_token=ctx.cancellation_token,
        )
        # Extract the final response from the output messages.
        final_response = output_messages[-1].content
        assert isinstance(final_response, str)
        return SearchResults(results=final_response)

class ToolInterventionHandler(DefaultInterventionHandler):
    async def on_send(self, message: Any, *, sender: AgentId | None, recipient: AgentId) -> Any | type[DropMessage]:
        if isinstance(message, FunctionCall):
            # Request user prompt for tool execution.
            user_input = input(
                f"Function call: {message.name}\nArguments: {message.arguments}\nDo you want to execute the tool? (y/n): "
            )
            if user_input.strip().lower() != "y":
                raise ToolException(content="User denied tool execution.", call_id=message.id)
        return message


async def main() -> None:
    model_client=AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )            


    work_dir = "/home/workspace" # tempfile.mkdtemp()

    # Create an local embedded runtime.
    runtime = SingleThreadedAgentRuntime(intervention_handlers=[ToolInterventionHandler()])

    async with DockerCommandLineCodeExecutor(
        image="scl2bp/local-develop:latest",
        container_name="dev-scl2bp",
        work_dir=work_dir,
        stop_container=False,
        auto_remove=False,
        ) as executor:


        # Register search tool agent.
        search_tool_agent = await ToolAgent.register(
            runtime,
            "tool_executor_agent",
            lambda: ToolAgent(
                description="Tool Executor Agent",
                tools=[google_search_tool],
            ),
        )

        # Register search agent.
        search_agent = await SearchAgent.register(
            runtime,
            "search_agent",
            lambda: SearchAgent(
                model_client,
                search_tool_agent.type,
            ),
        )

        # Register executor tool agent.
        executor_tool_agent = await Executor.register(
            runtime, "executor", 
            lambda: Executor(executor)
        )

        # Register the assistant and executor agents by providing
        # their agent types, the factory functions for creating instance and subscriptions.
        assistant_agent = await Assistant.register(
            runtime,
            "assistant",
            lambda: Assistant(
                model_client,
                executor_tool_agent.type
            ),
        )

        planer_agent = await PlanerAgent.register(
            runtime,
            "planner",
            lambda: PlanerAgent(
                model_client,
                search_agent.type,
                assistant_agent.type
            )
        )

        with open('core_coding.txt', 'w', encoding='utf-8') as sys.stdout:

            # Start the runtime and publish a message to the assistant.
            runtime.start()
            await runtime.publish_message(
                UserAssingment("""
                    Create a script that is able to modify the 'stock_returns_plot.py' file. 
                    Use the RedBaron library to interact with the python file.
                            """), 
                DefaultTopicId()
            )
            await runtime.stop_when_idle()

asyncio.run(main())