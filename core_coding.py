import os
import sys
import re
import asyncio

from dataclasses import dataclass
from typing import List

from autogen_core import (
    DefaultTopicId, 
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

# load .env file
from dotenv import load_dotenv
load_dotenv()

@dataclass
class Message:
    content: str


@default_subscription
class Assistant(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient) -> None:
        super().__init__("An assistant agent.")
        self._model_client = model_client
        self._chat_history: List[LLMMessage] = [
            SystemMessage(
                content="""
                Output shell command to read script files or to verify the existence of a file.
                Output python script to complete the task.
                Use existing python script as a starting point if requested.
                Use markdown block for shell commands and python scripts.
                The first line of the code block is as follows: '# filename: <filename>'.
                Work sequentially step by step through the task.
                Always provide only a single output for each step.
                Always save figures to file in the current directory. Do not use plt.show().
                """,
                # All code required to complete this task must be contained within a single response.
                # Use the 'coding_agent' conda environment.
                # Use the `conda run -n coding_agent` command to run the script or instal pip packages.

            )
        ]

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        self._chat_history.append(UserMessage(content=message.content, source="user"))
        result = await self._model_client.create(self._chat_history)
        print(f"\n{'-'*80}\nAssistant:\n{result.content}")
        self._chat_history.append(AssistantMessage(content=result.content, source="assistant"))  # type: ignore
        await self.publish_message(Message(content=result.content), DefaultTopicId())  # type: ignore


def extract_markdown_code_blocks(markdown_text: str) -> List[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)
    code_blocks: List[CodeBlock] = []
    for match in matches:
        language = match[0].strip() if match[0] else ""
        code_content = match[1]
        code_blocks.append(CodeBlock(code=code_content, language=language))
    return code_blocks


@default_subscription
class Executor(RoutedAgent):
    def __init__(self, code_executor: CodeExecutor) -> None:
        super().__init__("An executor agent.")
        self._code_executor = code_executor

    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext) -> None:
        code_blocks = extract_markdown_code_blocks(message.content)
        if code_blocks:
            result = await self._code_executor.execute_code_blocks(
                code_blocks, cancellation_token=ctx.cancellation_token
            )
            print(f"\n{'-'*80}\nExecutor:\n{result.output}")
            await self.publish_message(Message(content=result.output), DefaultTopicId())



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
    runtime = SingleThreadedAgentRuntime()

    async with DockerCommandLineCodeExecutor(
        image="scl2bp/local-develop:latest",
        container_name="dev-scl2bp",
        work_dir=work_dir,
        stop_container=False,
        auto_remove=False,


        ) as executor:  # type: ignore[syntax]
        # Register the assistant and executor agents by providing
        # their agent types, the factory functions for creating instance and subscriptions.
        await Assistant.register(
            runtime,
            "assistant",
            lambda: Assistant(
                model_client
            ),
        )
        await Executor.register(runtime, "executor", lambda: Executor(executor))

        with open('core_coding.txt', 'w', encoding='utf-8') as sys.stdout:

            # Start the runtime and publish a message to the assistant.
            runtime.start()
            await runtime.publish_message(
                Message("Create a script that is avle to modify the 'stock_returns_plot.py' file. Use the RedBaron library to interact with the python file."), DefaultTopicId()
            )
            await runtime.stop_when_idle()

asyncio.run(main())