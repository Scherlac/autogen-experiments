import os
import sys
import re
import asyncio
import json
from datetime import datetime

from dataclasses import dataclass
from typing import List, Any
import numpy as np

from autogen_core import (
    AgentId,
    AgentType,
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
    CreateResult,
    LLMMessage,
    SystemMessage,
    UserMessage,
)

from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
import tempfile

from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_core.tools import (
    FunctionTool,
    Tool,
    ToolSchema
)
from autogen_core.tool_agent import (
    ToolAgent, 
    ToolException, 
    tool_agent_caller_loop
)


# load .env file
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Message:
    content: str

@dataclass
class UserAssignment:
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
class Assignment:
    content: str

@dataclass
class AssignmentResults:
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
        search_agent_type: AgentType,
        assistant_agent_type: AgentType,
        description: str = "A planner agent", 
                 ) -> None:

        super().__init__(
            description)
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                You are a planning agent.
                The job is to break down complex tasks into smaller tasks.

""" f"""
                Your team members are:
                    Assistant agent (id: {assistant_agent_type.type}): Creates and executes scripts on local linux environment.
                    Search agent (id: {search_agent_type.type}): Searches for programming manuals on the internet.
""" """
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

                Manage the list of **collected information** as we get more results as follows:
                ```python
                # filename: collection.py
                collection = {
                    "<topic name>": {
                        "<sub topic name>": { # Only add sub topic for data from user message, search results or task results
                            "description": "<description>",
                            "usage": "<usage>", # Optional, only add if information is available
                            "file_name": "<file name>", # Optional, only add if information is available
                            "script": "<script>", # Optional, only add if information is available
                            "language": "<language>", # Optional, only add if information is available
                            "link": "<link>"  # Optional, only add if information is available
                        },
                        ...
                    },
                    ...
                }
                ```

                Manage the **tasks** and **prioritize** them based on the plan.
                Assign the next tasks using the following format:
                'TASK: <agent_id>, <task_id>, <task_description>'
                
                After all tasks are complete, summarize the findings and end with "TERMINATE".

                Store all important information in the file planer_state.json and use them to recover the state in case of failure.

                """,


                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.
                # You only plan the tasks and assign them to the correct agents.
                # Do not execute the tasks yourself. Do not write code to complete the tasks.

            )
        ]



    @message_handler
    async def handle_message(self, message: UserAssignment, ctx: MessageContext) -> None:
        print(f"\n{'-'*80}\nAssignment:\n{message.content}", flush=True)
        self._chat_history.append(UserMessage(content=message.content, source="user"))

        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nPlaner:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="planner"))

        await self.publish_message(Task(content=result.content), DefaultTopicId())


    @message_handler
    async def handle_taskresult(self, message: TaskResults, ctx: MessageContext) -> None:
        # print(f"\n{'-'*80}\nTask Results:\n{message.results}", flush=True)
        self._chat_history.append(AssistantMessage(content=message.results, source="task_handler"))

        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nPlaner:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="planner"))

        await self.publish_message(Task(content=result.content), DefaultTopicId())


@default_subscription
class TaskHandlerAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        search_agent_type: AgentType,
        assistant_agent_type: AgentType,
        description: str = "A task handler agent.",
                 ) -> None:

        super().__init__(
            description)
        self._model_client = model_client
        self._search_agent_type = search_agent_type
        self._assistant_agent_type = assistant_agent_type
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content = """
                You are a task handler agent.
                Planer agent will provide you details about the main objective, collected information, the plan.
                Planer agent will prepare ideas of multiple tasks as follows:
                'TASK: <agent_id>, <task_id>, <task_description>'
""" f"""
                You have two team members:
                    Search agent (id: {search_agent_type.type}): Searches for information on internet.
                    Assistant agent (id: {assistant_agent_type.type}): Creates and executes scripts on local environment.
""" """
                Your job is to prepare a single task and assign to the specific agents.
                Specific agents have no access to chat history. So, provide all related context
                and details for them based on chat history.

                Output a single task you intend to assign as follows:
                'TASK: <agent_id>, <task_id>, <task description>, <context and details>'

                Output 'TERMINATE' if all tasks are completed or the planer agent ends the conversation with 'TERMINATE'.

                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.

            )    
        ]

    async def proces_task(self):
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nTask Handler:\n{result.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=result.content, source="task_handler"))

        tasks = []
        # split the result into lines and process each line
        for message in result.content.split("\n"):
            task_rgex = r".*TASK: (\w+), (\w+), (.+)" 
            if re.match(task_rgex, message):
                agent_id, task_id, task_description = re.match(task_rgex, message).groups()
                # In case agent_id is 'similar' to search_agent_type, send the task to the search agent.
                if agent_id == self._search_agent_type.type:
                    agent_id = f"Search Agent - Task {task_id}"
                    recipient = AgentId(self._search_agent_type, agent_id)
                    tasks.append({"recipient": recipient, "task_id": task_id, 
                                  "task_description": task_description,
                                  "agent_id": agent_id,
                                  "message": Search(query=task_description)})
                # In case agent_id is 'similar' to assistant_agent_type, send the task to the assistant agent.
                elif agent_id == self._assistant_agent_type.type:
                    agent_id= f"Assistant Agent - {task_id}"
                    recipient = AgentId(self._assistant_agent_type, agent_id)
                    tasks.append({"recipient": recipient, "task_id": task_id, 
                                  "task_description": task_description,
                                  "agent_id": agent_id,
                                  "message": Assignment(content=task_description)})
                else:
                    raise ValueError(f"Unknown agent_id: {agent_id}")

        if len(tasks) > 0:
            tasks = [ { 
                "agent_id": task["agent_id"],
                "task_id": task["task_id"],
                "task_description": task["task_description"],
                "result": self.send_message(task["message"], task["recipient"])
                }
                for task in tasks]

            for task in tasks:
                result = await task["result"]
                task.pop("result")
                if isinstance(result, SearchResults):
                    agent = "search_agent"
                    content = result.results
                elif isinstance(result, TaskResults):

                    content = result.results
                elif isinstance(result, AssignmentResults):
                    agent = "assistant_agent"
                    content = result.results
                else:
                    raise ValueError(f"Unknown result type: {result}")
                task['content'] = content

                print(f"\n{'-'*80}\n{task['agent_id']}:\n{task['task_description']}\n{content}", flush=True)
                self._chat_history.append(AssistantMessage(content=content, source=agent))

            return await self.publish_message(
                TaskResults(results=f"Report on tasks:\n```json\n{"\n```\n```json\n".join([f"{
                    json.dumps(task)
                    }" for task in tasks])}\njson\nContinuing with planing activities."), DefaultTopicId())
        
        if "TERMINATE" not in result.content and \
            "SUCCESS" not in result.content and \
            "FAILURE" not in result.content:
            return await self.publish_message(TaskResults(results="Please provide a valid task or end the conversation with 'TERMINATE'."), DefaultTopicId())

    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> None:
        # print(f"\n{'-'*80}\nTask:\n{message.content}", flush=True)
        self._chat_history.append(AssistantMessage(content=message.content, source="task_handler"))

        await self.proces_task()

@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, 
            model_client: ChatCompletionClient,
            tool_agent_type: AgentType,
            description: str = "An assistant agent.",
            ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
You are a coding assistant agent.
Your job is to create script (bash, pwsh script or python code) to work on the assigned task and then provide final report on the given task.

You only have an interactive chat with an executor agent with a local linux environment.
The executor agent has the current directory as workspace.
In case exact facts are not available, you can return FAILURE and ask for more information.



# Working on the task (first phase):

Use the following rules to create script and work the task:
- Provide only a single output of maximum 200 lines.
- Add logging (print or echo) and try-except block to track and debug the script.
- Use '<msg> Done', '<msg> Failed', 'Error: <msg>' tracking messages to indicate the completion of the task.
- Always save figures to file. Do not use show() or display() commands.
- Output markdown script (bash, pwsh or python) to create a file and execute it as follows:
```<language>
# filename: <name of the script> # Optional, add only to retain the final script file for future reference.
# description: <description of the script>
# usage: <usage of the script>
<code>

# Final report (second phase):

Use the following rules and keywords only to report your final results, do not use them in the script:
- Output 'SUCCESS: <name of the script>, <description of the script> <usage of the script>' if executor was successful and the task is completed.
- Also output 'SUMMARY: <summary of the task>' if the task is completed.
- Output 'FAILURE: <reason for failure>, <additional requirements>, <search recommendation>' if unable to complete the task or to ask for more information.

```

                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.
                # Output SEARCH: 'query text' if anditional information is required.
                # The first line of the code block is as follows: '# filename: <name of the script>'.
                # Use existing script file as a starting point if requested.

            )
        ]
        self._tool_agent_type = tool_agent_type
        self._tool_agent = AgentId(tool_agent_type, f"Tool Agent - {self.id.key}")

    @message_handler
    async def handle_task(self, message: Assignment, ctx: MessageContext) -> AssignmentResults:
        session: List[LLMMessage] = [
            self._chat_history[0],
            UserMessage(content=message.content, source="user")
        ]
        result = await self._model_client.create(session)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}", flush=True)
        session.append(AssistantMessage(content=result.content, source="assistant"))

        for i in range(5):
            message = await self.send_message(CommandMessage(command=result.content), self._tool_agent)
            session.append(AssistantMessage(content=message.output, source="executor"))

            result = await self._model_client.create(session)
            print(f"\n{'-'*80}\nAssistant:\n{result.content}", flush=True)
            if "SUCCESS" in result.content or "FAILURE" in result.content:
                return AssignmentResults(results=result.content)

        return AssignmentResults(results="The task is not completed. Please provide more information.")

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
    async def handle_commandmessaga(self, message: CommandMessage, ctx: MessageContext) -> CommandResponse:
        code_blocks = extract_markdown_code_blocks(message.command)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            print(f"\n{'-'*80}\nExecutor:\n{result.output}", flush=True)
            return CommandResponse(output=result.output)
        return CommandResponse(output="No code blocks found in the message.")




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
        tool_schema: List[ToolSchema],
        tool_agent_type: AgentType,
        description: str = "Search Agent",
                 ) -> None:
        super().__init__(description)
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
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(type=tool_agent_type, key=self.id.key)

        #self._tools = tools

    @message_handler
    async def handle_message(self, message: Search, ctx: MessageContext) -> SearchResults:
        session: List[LLMMessage] = [
            self._chat_history[0],
            UserMessage(content=message.query, source="user")]
        output_messages = await tool_agent_caller_loop(
            self,
            tool_agent_id=self._tool_agent_id,
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
        # if isinstance(message, FunctionCall):
        #     # Request user prompt for tool execution.
        #     user_input = input(
        #         f"Function call: {message.name}\nArguments: {message.arguments}\nDo you want to execute the tool? (y/n): "
        #     )
        #     if user_input.strip().lower() != "y":
        #         raise ToolException(content="User denied tool execution.", call_id=message.id)
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
            "search_tool_agent",
            lambda: ToolAgent(
                description="Search Tool Agent",
                tools=[google_search_tool],
            ),
        )

        # Register search agent.
        search_agent = await SearchAgent.register(
            runtime,
            "search_agent",
            lambda: SearchAgent(
                description="Search Agent",
                model_client=model_client,
                tool_schema=[google_search_tool.schema],
                tool_agent_type=search_tool_agent
                
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
            "assistant_agent",
            lambda: Assistant(
                description="An assistant agent.",
                model_client=model_client,
                tool_agent_type=executor_tool_agent
            ),
        )

        planer_agent = await PlanerAgent.register(
            runtime,
            "planner",
            lambda: PlanerAgent(
                description="A planner agent", 
                model_client=model_client,
                search_agent_type=search_agent,
                assistant_agent_type=assistant_agent
                
            )
        )

        task_handler_agent = await TaskHandlerAgent.register(
            runtime,
            "task_handler",
            lambda: TaskHandlerAgent(
                description="A task handler agent.",
                model_client=model_client,
                search_agent_type=search_agent,
                assistant_agent_type=assistant_agent
            )
        )

        # The output is written in a file named as this script file with date and time appended to it with .txt extension.
        filename = os.path.basename(__file__).replace(".py", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filename, 'w', encoding='utf-8') as sys.stdout:

            # Start the runtime and publish a message to the assistant.
            runtime.start()
            # await runtime.publish_message(
            #     UserAssignment("""
            #         Create a script called 'refactor.py' that is able to modify the 'stock_returns_plot.py' file. 
                    
            #         The "refactor.py' script should:
            #             - Use the RedBaron library to interact with the python file.
            #             - Replace hard coded values with variables.
            #             - Refactor the code to use functions.

            #         The stock_returns_plot.py file is in current folder and it is version controlled, we can revert to the original file if needed.
                               
            #         How to find local files on linux?
            #         Where is the 'stock_returns_plot.py' file located?
            #         How to use git to revert the local changes of a file?
            #         Can we use RedBaron to:
            #         - find and list the hard coded values in the file?
            #         - replace the hard coded values with variables?
            #         - refactor the code to use functions?
            #         What is the purpose of the 'stock_returns_plot.py' file?
            #         How to add command line arguments to 'stock_returns_plot.py'?
                    
            #         Where is the 'refactor.py' script located?
            #         What is already done in the 'refactor.py' script?
            #         How to use it to refactor the 'stock_returns_plot.py' file?
                               
            #                 """), 
            #     DefaultTopicId()
            # )

#             await runtime.publish_message(
#                 UserAssignment( \
# """
# Similar projects can be found at: https://github.com/Scherlac/autogen-experiments/blob/main/ext_coding.py
# Download the mentioned file and check the code for autogen imports.
# Coder agent is able to use curl to download the file from the given URL.

# Ensure the coder also uses the autogen library to implement the sceptic agent. Eg `grep -A 30 'import'` the code for autogen imports.
# Give an example of how to use the sceptic agent and how to test it. 

# Model client:                      
# ```python
# model_client=AzureOpenAIChatCompletionClient(
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
#         model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#         temperature=0.0,
#     )            
# ```

# Register the agent as follows:
# ```python
# planer_agent = await ScepticAgent.register(
#     runtime,
#     "sceptic",
#     lambda: ScepticAgent(
#         description="A sceptic agent", 
#         model_client=model_client,
#     )
# )
# ```  

# Create runtime as follows:
# ```python
# runtime = SingleThreadedAgentRuntime()
# ```

# Start the runtime and publish a message to the assistant as follows:
# ```python
# runtime.start()
# await runtime.publish_message(
#     UserAssignment("Please write a hello world program in python."), 
#     DefaultTopicId() )
# ```

# Main contains the following code to start:
# ```python
# asyncio.run(main())
# ```

# """), 
#                 DefaultTopicId()
#             )
#             await runtime.publish_message(
#                 UserAssignment( \
# """
# Download the following script: https://github.com/Scherlac/autogen-experiments/blob/main/ext_coding.py
# Review the code and make a plan to refactor the code.

# """), 
#                 DefaultTopicId()
#             )

            await runtime.publish_message(
                UserAssignment( \
"""
Create a fact dictionary with embeddings that can hold searchable information on content of local files.
Download the following script: https://github.com/Scherlac/autogen-experiments/blob/main/ext_coding.py
Fill the fact dictionary with class information about the downloaded script.

You already started but not really finished. List the content of the working directory and check the content.
The fact_dictionary.json still has no information on the downloaded script.

"""), 
                DefaultTopicId()
            )

            # # wait for 20 minutes for the agents to complete the tasks
            # await asyncio.sleep(1200)
            await runtime.stop_when_idle()

asyncio.run(main())