import os
import sys
import re
import asyncio
import json
import random
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
from autogen_ext.teams.magentic_one import MagenticOne

    


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

# read planer state from file
def read_planer_state():
    """
    Read the planer state from the file planer_state.json.

    Returns:
        dict: The planer state.
    """
    if not os.path.exists("planer_state.json"):
        return {}
    with open("planer_state.json", "r") as file:
        return json.load(file)
    

# write planer state to file
def write_planer_state(state: dict):
    """
    Write the planer state to the file planer_state.json.

    Args:
        state (dict): The planer state.
    """
    with open("planer_state.json", "w") as file:
        json.dump(state, file)

# function tool for reading and writing planer state
read_planer_state_tool = FunctionTool(read_planer_state, description="Read the planer state from the file planer_state.json.")
write_planer_state_tool = FunctionTool(write_planer_state, description="Write the planer state to the file planer_state.json.")

@default_subscription
class PlanStoreAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        tool_schema: List[ToolSchema],
        tool_agent_type: AgentType,
        description: str = "Plan Store Agent",
                 ) -> None:
        super().__init__(description)
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                You are a plan store agent.
                Your tools are read_planer_state_tool and write_planer_state_tool.
                You make only one action call at a time.
                Once you have the results, you send it to the chat, never do other actions based on them.
                """,
            )
        ]
        self._tool_schema = tool_schema
        self._tool_agent_id = AgentId(type=tool_agent_type, key=self.id.key)

        #self._tools = tools

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> Message:
        session: List[LLMMessage] = [
            self._chat_history[0],
            UserMessage(content=message.content, source="user")]
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
        return Message(content=final_response)



@default_subscription
class PlanerAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        # planer_status_agent_type: AgentType,
        # search_agent_type: AgentType,
        # assistant_agent_type: AgentType,
        task_handler_agent_type: AgentType,
        description: str = "A planner agent", 

                 ) -> None:

        super().__init__(
            description)
        self._model_client = model_client
        self._task_handler_agent_type = task_handler_agent_type

        # self._planer_status_agent_type = planer_status_agent_type
        # self._planer_status_agent = AgentId(planer_status_agent_type, f"Planer Status Agent - {self.id.key}")
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                You are a planning agent. You will get an assignment.
                You will work in multiple turns, in each turn you will manage the plan, collected information and assign tasks.


""" f"""
                Your have following agents available:
                    Task handler agent (id: {task_handler_agent_type.type}): Assigns tasks to the specific agents. 
""" """

                List the original assignment as follows:
                ```text
                # Assignment:
                <original assignment>
                ```

                Manage the **plan** by creating, updating, refining and following it to accomplish the task as follows:
                ```text
                # Plan:
                - step <step number>: <step description>
                ```

                Manage the list of **collected information** by collecting topics and subtopics as we progress as follows:
                ```text
                # Collected Information:

                ## <topic name>
                <description of the topic>
                
                ### <sub topic name> // Only add sub topic for data from 'user message', 'search results' or 'task results'
                <description of the sub topic>

                - usage: "<usage>", # Optional, only add if information is available
                - file_name: "<file name>", # Optional, only add if information is available
                - script: "<script>", # Optional, only add if information is available
                - language: "<language>", # Optional, only add if information is available
                - link: "<link>"  # Optional, only add if information is available

                ```

                Manage the **tasks** and **prioritize** them by output a maximum of three task at a time, eg. task id 1-3. than later id 4-6. Assign the next tasks using the following format:
                'TASK: <agent_id>, <task_id>, Todo: <task description>, Data: <parameters, data, url, etc.>, Background: <context and details>, Output: <ask to output the required information and the proof of completion>'
                Once the selected task is completed you will get the results and you review te plan and assign the next task.
                 
                After tasks are complete, review the plane, create further task brake down and update the task if needed. 
                Task brake down should be done by the planer agent. Task agent may request further brake down if needed.

                Output 'TERMINATE' only if no more steps or task to be processed and all results, reports and reviews are available and final report is ready to be created.

                """,


                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.
                # You only plan the tasks and assign them to the correct agents.
                # Do not execute the tasks yourself. Do not write code to complete the tasks.

            )

        ]

    async def process_plan(self, message: str):
        self._chat_history.append(UserMessage(content=f"USER MESSAGE:\n{message}", source="user"))

        status = "TIMEOUT"

        while True:

            result = await self._model_client.create(self._chat_history)
            print(f"\n{'-'*80}\nPlaner ({self._id}):\n{result.content}", flush=True)
            self._chat_history.append(AssistantMessage(content=result.content, source="planner"))

            if "TERMINATE" in result.content:
                status = "COMPLETE"
                break
            
            # find the first line starts with 'TASK:'
            lines = result.content.split("\n")
            task_rgex = r"^[- \t]*TASK: (?P<agent_id>\w+), (?P<task_id>\w+), (?P<task_description>.+)"
            tasks = [line for line in lines if re.match(task_rgex, line)]
            
            if len(tasks) == 0:
                self._chat_history.append(SystemMessage(content=f"**system message**:\nNO TASK FOUND\nPlease assign a task OR terminate.", ))
                continue

            first_task = next(iter(tasks))

            match = re.match(task_rgex, first_task)
            agent_id, task_id, task_description = match.groups()
            agent_id = f"{self._id} -> T:{task_id} -> Task"

            # get a new agent for each task
            task_handler_agent = AgentId(self._task_handler_agent_type, agent_id)

            task_result = await self.send_message(Task(content=result.content), task_handler_agent)
            self._chat_history.append(UserMessage(
                    content=f"TASK HANDLER:\n{task_result.results}\n\nUpdate plane or select new task or terminate?", 
                    source="task_handler"))

        self._chat_history.append(UserMessage(
                content=f"USER MESSAGE:\n\nTERMINATING ({status})\n\nPlease summarize our discussion, focus on the current task, pay attention to include all important details.", 
                source="user"))
        report = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nPlaner ({self._id}) - Final report:\n{report.content}", flush=True)
        return AssignmentResults(results = f"RESULT:\n{result.content}\n\nREPORT:\n{report.content}")


    @message_handler
    async def handle_message(self, message: UserAssignment, ctx: MessageContext) -> None:
        print(f"\n{'-'*80}\nPlaner ({self._id}) - User assignment:\n{message.content}", flush=True)
        await self.process_plan(message.content)


    @message_handler
    async def handle_task(self, message: Assignment, ctx: MessageContext) -> AssignmentResults:
        print(f"\n{'-'*80}\nPlaner ({self._id}) - Assignment:\n{message.content}", flush=True)
        return await self.process_plan(message.content)


@default_subscription
class TaskHandlerAgent(RoutedAgent):
    def __init__(self, 
        model_client: ChatCompletionClient,
        search_agent_type: AgentType,
        assistant_agent_type: AgentType,
        planar_agent_type: AgentType,
        description: str = "A task handler agent.",
                 ) -> None:

        super().__init__(
            description)
        self._model_client = model_client
        self._search_agent_type = search_agent_type
        self._assistant_agent_type = assistant_agent_type
        self._planar_agent_type = planar_agent_type
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content = """
                You are a task handler agent. You are responsible for assigning tasks to specific agents based assigned task and chat history.
                'TASK: <agent_id>, <task_id>, <task_description>'
""" f"""
                You have following agents available:
                    - Assistant agent (id: {assistant_agent_type.type}): Creates and executes scripts on linux consol with internet access (bash, pwsh, python, ls, git, curl, etc.). Versatile in capabilities.
                    - Search agent (id: {search_agent_type.type}): Only able to search information on Google not able to download files. Really limited in capabilities.
                    - Planer agent: (id: {planar_agent_type.type}): Breaks down complex tasks into smaller tasks and assigns them to further agents. Really expensive to use.

""" """
                Your job is to prepare a single task based on the chat history and assign it to the specific agent.
                The task should be assigned to the agent that is most suitable for the task.
                The agents are only able to communicate with you and not aware of each other and not aware of the chat history.

                Output a single task you intend to assign to an agent in the following format:
                'TASK: <agent_id>, <task_id>, Todo: <task description>, Data: <parameters, data, url, etc.>, Background: <context and details>, Output: <ask to output the required information and the proof of completion>'

                In case the selected agent fails, repeat the task with more information or you can try to assign to another agent.
                If the task is too complex, try to assign it to the planer agent. If the planer agent is too busy, try to assign it to the assistant agent.
                
                Output 'TERMINATE' if all task are processed and the final report is ready to be created.
                Output 'SUMMARY' on 'TERMINATE' with the summary of the task result as follows:
                'SUMMARY: Agent: <specific agent>, Status: <status of the task>, Summary: <summary of the task>, Result: <result of the task>, Usage: <usage of the script>, Output: <summary of the script output>', Findings: <findings of the task>
                Output 'OUTPUT' on 'SUCCESS' with the detailed output of the task as follows:
                'OUTPUT:
                ```<format>
                <output>
                ```'



                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.

            )    
        ]

    async def process_task(self, message: str):
        self._chat_history.append(AssistantMessage(content=message, source="user"))

        status = "TIMEOUT"

        while True:
            result = await self._model_client.create(self._chat_history)
            print(f"\n{'-'*80}\nTask Handler ({self._id}):\n{result.content}", flush=True)
            self._chat_history.append(AssistantMessage(content=result.content, source="task_handler"))

            if "TERMINATE" in result.content:
                status = "COMPLETE"
                break
            
            if "SUCCESS" in result.content or "SUMMARY" in result.content:
                status = "SUCCESS"
                break
                
            if "FAILURE" in result.content:
                status = "FAILURE"
                break
                #return await self.publish_message(TaskResults(results=result.content), DefaultTopicId())


            tasks = []
            # split the result into lines and process each line
            for message in result.content.split("\n"):
                task_rgex = r".*TASK: (\w+), (\w+), (.+)" 
                if re.match(task_rgex, message):
                    agent_id, task_id, task_description = re.match(task_rgex, message).groups()

                    # In case agent_id is 'similar' to search_agent_type, send the task to the search agent.
                    if agent_id == self._search_agent_type.type:
                        agent_id = f"{self._id} -> T:{task_id} -> Search"
                        recipient = AgentId(self._search_agent_type, agent_id)
                        tasks.append({"recipient": recipient, "task_id": task_id, 
                                    "task_description": task_description,
                                    "agent_id": agent_id,
                                    "agent": "search_agent",
                                    "message": Search(query=task_description)})
                    # In case agent_id is 'similar' to assistant_agent_type, send the task to the assistant agent.
                    elif agent_id == self._assistant_agent_type.type:
                        agent_id= f"{self._id} -> T:{task_id} -> Assistant"
                        recipient = AgentId(self._assistant_agent_type, agent_id)
                        tasks.append({"recipient": recipient, "task_id": task_id, 
                                    "task_description": task_description,
                                    "agent_id": agent_id,
                                    "agent": "assistant_agent",
                                    "message": Assignment(content=task_description)})
                    # In case agent_id is 'similar' to planar_agent_type, send the task to the planer agent.
                    elif agent_id == self._planar_agent_type.type:
                        # reduce chance to send the task to the planer agent in case we already have a lot of planers
                        if False: #random.random() < (40 / len(str(self._id))):
                            agent_id= f"{self._id} -> T:{task_id} -> Planer"
                            recipient = AgentId(self._planar_agent_type, agent_id)
                            tasks.append({"recipient": recipient, "task_id": task_id, 
                                        "task_description": task_description,
                                        "agent_id": agent_id,
                                        "agent": "planer_agent",
                                        "message": Assignment(content=task_description)})
                        else:
                            self._chat_history.append(UserMessage(content="Instead of starting new sub project please terminate and report back to main planer and request further task breakdown on this task.", source="planer_agent"))
                            print(f"\n{'-'*80}\nTask Handler ({self._id}) - Planer agent too busy.", flush=True)


                    else:
                        raise ValueError(f"Unknown agent_id: {agent_id}")

            if len(tasks) > 0:
                tasks = [ { 
                    "agent_id": task["agent_id"],
                    "agent": task["agent"],
                    "task_id": task["task_id"],
                    "task_description": task["task_description"],
                    "result": self.send_message(task["message"], task["recipient"])
                    }
                    for task in tasks]

                for task in tasks:
                    agent_result = await task["result"]
                    task.pop("result")
                    if isinstance(agent_result, SearchResults):
                        agent = "search_agent"
                        content = agent_result.results
                    elif isinstance(agent_result, TaskResults):

                        content = agent_result.results
                    elif isinstance(agent_result, AssignmentResults):
                        agent = "assistant_agent"
                        content = agent_result.results
                    else:
                        raise ValueError(f"Unknown result type: {agent_result}")
                    task['content'] = content

                    # print(f"\n{'-'*80}\nTask Handler ({self._id})\nANSWER FROM: {task['agent_id']}:\nTASK: {task['task_description']}\nCONTENT: {content}", flush=True)
                    self._chat_history.append(UserMessage(
                                content= f"ANSWER FROM: {task['agent']}:\nCONTENT: {content}\n\nProvide more data and try again or continue, terminate and report back to planer?",
                                source=task['agent']
                                ))
                                               

            # else:
            #     break
#        self._chat_history.append(UserMessage(content=f"USER MESSAGE:\nTERMINATING\nPlease summarize our discussion, pay attention to include all important details and data related to main assignment.", source="user"))
        self._chat_history.append(UserMessage(
                content=f"USER MESSAGE:\nTermination with status {status} accepted.\n\nPlease summarize all processed task and there status, results, created script, usage, outputs and findings.", 
                source="user"))
        report = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nTask Handler ({self._id}) - Final report:\n{report.content}", flush=True)
        return TaskResults(results = f"RESULT:\n{result.content}\n\nREPORT:\n{report.content}")


    @message_handler
    async def handle_task(self, message: Task, ctx: MessageContext) -> TaskResults:
        # print(f"\n{'-'*80}\nTask:\n{message.content}", flush=True)

        return await self.process_task(message.content)

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
Your job is to:
- create script (bash, pwsh script or python code) to work on the assigned task or
- process the output of the executor agent and decide on the next step eg. retry and correct the script, finish 
- create the final report when you are done, eg. the script finishes correctly.

You work on the task, while you output code block. The system may limit the attempts to complete the task (RETRY COUNTS).
You only have an interactive chat with an executor agent with a local linux environment.
The executor agent has the current directory as workspace. It will output the first 200 lines of the script output.
In case exact facts or  input parameters are not available, you should output 'FAILURE' and ask for more information. The executor agent will not provide parameters.


# Working on the task (ASSIGNMENT, EXECUTOR):

Use the following rules to create script and work the task:
- Provide only a single output of maximum 200 lines.
- Log to console (print or echo) and add try-except block to track and debug the script.
- Use '<msg> Done', '<msg> Failed', 'Error: <msg>' tracking messages to indicate the completion of the task.
- Use f"Success: <msg with formatted outputs>" tracking message to indicate the success run of the script.
- Always save figures to file. Do not use show() or display() commands.
- Output markdown script (bash, pwsh or python) to create a file and execute it as follows:

```<language>
# filename: <name of the script> # Optional, add only to retain the final script file for future reference.
# description: <description of the script>
# usage: <usage of the script>

<code>
```

# Final report (final phase, Done, TERMINATING):

Use the following rules and keywords only to report your final results, do not use them in the script:
- Output 'SUCCESS: <name of the script>, <description of the script> <usage of the script>' if executor was successful and the task is completed.
- Also output 'SUMMARY: <summary of the task>, <summary of the script output>' if the task is completed.
- Also output 'OUTPUT: <output of the script>' if the task is completed.
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
        self._chat_history.append(UserMessage(content=f"USER MESSAGE:\nASSIGNMENT:\n{message.content}", source="user"))

        trys: int = 8

        status = "TIMEOUT"

        while trys > 0:

            # self._chat_history.append(SystemMessage(content=f"**system message**:\nRETRY COUNTS: {trys}"))
            trys -= 1

            result = await self._model_client.create(self._chat_history)
            print(f"\n{'-'*80}\nAssistant ({self._id}):\n{result.content}", flush=True)
            self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))

            if "SUCCESS" in result.content:
                status = "SUCCESS"
                break
            if "FAILURE" in result.content:
                status = "FAILURE"
                break

            message = await self.send_message(CommandMessage(command=result.content), self._tool_agent)
            self._chat_history.append(UserMessage(
                    content=f"EXECUTOR OUTPUT:\n{message.output}\n\nRemaining retry count: {trys}\n\nReview the output and refine the script or summarize and terminate.", 
                    source="executor"))

            # if "Success" in message.output:
            #     status = "SUCCESS"
            #     break


        # self._chat_history.append(UserMessage(content=f"USER MESSAGE:\nTERMINATING ({status})\nPlease summarize our discussion, pay attention to include all important details and data related to main assignment.", source="user"))
        self._chat_history.append(UserMessage(content=f"USER MESSAGE:\nTermination with status {status} accepted.\nPlease summarize our discussion, focusing on the current assignment, pay attention to include all important details.", source="user"))

        report = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant ({self._id}) - Final report:\n{report.content}", flush=True)
        return AssignmentResults(results = f"RESULT:\n{result.content}\n\nREPORT:\n{report.content}")

def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    # pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    # extract the language and code content from the markdown text
    pattern = re.compile(r"```(?P<language>\n?[\w\+\-]+)?\n(?P<code>[\s\S]*?)```")
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
    async def handle_command_message(self, message: CommandMessage, ctx: MessageContext) -> CommandResponse:
        code_blocks = extract_markdown_code_blocks(message.command)
        if code_blocks:
            try:
                result = await self._code_executor.execute_code_blocks(
                    code_blocks, cancellation_token=ctx.cancellation_token
                )
                # truncate the output to 200 lines
                lines = result.output.split("\n")
                result.output = "\n".join(lines[:200])

                if len(lines) > 200:
                    # add a message to indicate the output is truncated
                    result.output += "\n{'-'*50}\nOutput truncated to 200 lines."


                print(f"\n{'-'*80}\nExecutor:\n{result.output}", flush=True)
                return CommandResponse(output=result.output)
            except Exception as e:
                print(f"\n{'-'*80}\nExecutor:\nExecution error: {str(e)}", flush=True)
                return CommandResponse(output=f"Execution error: {str(e)}")
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
        print(f"\n{'-'*80}\nSearch Agent ({self._id}):\n{final_response}", flush=True)
        assert isinstance(final_response, str)
        return SearchResults(results=final_response)

# FIXME The api has been changed 
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

class MyMagentic(MagenticOne):
    
    async def register_agents(self, reg_fn):
        return reg_fn(self._runtime)
        


async def main() -> None:



    model_client=AzureOpenAIChatCompletionClient(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )            


    async with DockerCommandLineCodeExecutor(
        image="pscrdevops210e54.azurecr.io/scl2bp/nvidia-2204:dev-latest",
        container_name="dev-scl2bp-agent",
        work_dir="/workspace/agent-workspace",
        stop_container=False,
        auto_remove=False,
        ) as executor:


        from autogen_ext.teams.magentic_one import MagenticOne

        m1 = MyMagentic(client=model_client,
                code_executor=executor,
                hil_mode=True,
                        )


        # Register search tool agent.
        search_tool_agent = await m1.register_agents(lambda runtime : ToolAgent.register(
            runtime,
            "search_tool_agent",
            lambda: ToolAgent(
                description="Search Tool Agent",
                tools=[google_search_tool],
            ),
        ) )

        # Register search agent.
        search_agent = await m1.register_agents(lambda runtime : SearchAgent.register(
            runtime,
            "search_agent",
            lambda: SearchAgent(
                description="Search Agent",
                model_client=model_client,
                tool_schema=[google_search_tool.schema],
                tool_agent_type=search_tool_agent
                
            ),
        ))

        # Register executor tool agent.
        executor_tool_agent = await m1.register_agents(lambda runtime : Executor.register(
            runtime, "executor", 
            lambda: Executor(executor)
        ) )

        # Register the assistant and executor agents by providing
        # their agent types, the factory functions for creating instance and subscriptions.
        assistant_agent = await m1.register_agents(lambda runtime : Assistant.register(
            runtime,
            "assistant_agent",
            lambda: Assistant(
                description="An assistant agent.",
                model_client=model_client,
                tool_agent_type=executor_tool_agent
            ),
        ) )

        # planer_tool_agent = await ToolAgent.register(
        #     runtime,
        #     "planer_tool_agent",
        #     lambda: ToolAgent(
        #         description="Planer Status Agent",
        #         tools=[read_planer_state_tool, write_planer_state_tool],
        #     )
        # )

        # planer_status_agent = await PlanStoreAgent.register(
        #     runtime,
        #     "planer_status_agent",
        #     lambda: PlanStoreAgent(
        #         description="Plan Store Agent",
        #         model_client=model_client,
        #         tool_schema=[read_planer_state_tool.schema, write_planer_state_tool.schema],
        #         tool_agent_type=planer_tool_agent
        #     )
        # )


        planer_agent = await m1.register_agents(lambda runtime : PlanerAgent.register(
            runtime,
            "planner",
            lambda: PlanerAgent(
                description="A planner agent", 
                model_client=model_client,
                task_handler_agent_type=task_handler_agent
                # planer_status_agent_type=planer_status_agent,
                # search_agent_type=search_agent,
                # assistant_agent_type=assistant_agent
                
            )
        ) )

        task_handler_agent = await m1.register_agents(lambda runtime : TaskHandlerAgent.register(
            runtime,
            "task_handler",
            lambda: TaskHandlerAgent(
                description="A task handler agent.",
                model_client=model_client,
                search_agent_type=search_agent,
                assistant_agent_type=assistant_agent,
                planar_agent_type=planer_agent

            )
        )
        )
        # The output is written in a file named as this script file with date and time appended to it with .txt extension.
        filename = os.path.basename(__file__).replace(".py", f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        with open(filename, 'w', encoding='utf-8') as sys.stdout:

            task = \
"""

# Summary of Previous Task:

## Task Overview
The primary task was to determine how to query Visual Studio Code (VS Code) using UNIX socket connections and to understand the purpose of those sockets.

## Steps Taken
1. **Initial Research**: The task began with executing commands to identify running processes related to VS Code and the ports they were using. This included the `ps aux`, `ss -a --unix -p`, and `netstat -aenpx` commands.
2. **Findings**: It was confirmed that Visual Studio Code was running on the local machine, and several UNIX socket connections were identified, which are utilized by VS Code for various operations.

## Key Findings
- **UNIX Socket Connections Identified**:
  - `/tmp/vscode-remote-containers-ipc-19173b3a-fdbb-4f43-8971-7bab47795c59.sock`: IPC for remote containers.
  - `/tmp/vscode-ipc-91664a04-6b47-4607-ba1b-15be32397d2c.sock`: General IPC within VS Code.
  - `/tmp/vscode-git-8743d884f6.sock`: Associated with Git operations.
  - `/tmp/vscode-ipc-92e2ba10-075e-45e4-a0b2-7ca90b0609c3.sock`: IPC for internal components.
  - `/tmp/vscode-remote-containers-ipc-764c1760-9c3b-47fe-ae96-209fddda50fd.sock`: Another IPC for remote containers.
  - `/tmp/vscode-ipc-7e57ba78-bb1e-4f54-a6cc-54b1201fa242.sock`: IPC for extensions and internal services.

## Methods to Query VS Code using UNIX Sockets
- Remote Development Tips from the official documentation.
- Extensions like Socket.io client for VSCode.
- Information on UNIX domain sockets for inter-process communication.
- VS Code Extension API for creating custom solutions.
- Discussions on common socket issues on platforms like Stack Overflow.

## Conclusion
The task was successfully completed, providing a comprehensive overview of how to query Visual Studio Code using UNIX socket connections and detailing the purpose of each identified socket. This information is crucial for troubleshooting and optimizing the performance of Visual Studio Code, especially in remote development and version control scenarios.

The task has been terminated as all objectives were met, and the findings were compiled into a final report.

----

# New Task:

[Language Server Extension Guide](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide)

Make a connection to the language server using UNIX socket connections on local machine.

"""
            from autogen_agentchat.ui import Console
            result = await Console( m1.run_stream(task=task)) 
            print(result, flush=True)


asyncio.run(main())