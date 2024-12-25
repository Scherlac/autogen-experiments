
from typing import Callable, List, Sequence
from autogen_core.models import ChatCompletionClient
from autogen_agentchat.base import TerminationCondition, ChatAgent
from autogen_agentchat.messages import (
    AgentMessage,
)
from autogen_agentchat.teams import BaseGroupChat, SelectorGroupChatManager, BaseGroupChatManager

class CustomGroupChatManager(SelectorGroupChatManager):
    def __init__(
        self,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
        model_client: ChatCompletionClient,
        selector_prompt: str,
        allow_repeated_speaker: bool,
        selector_func: Callable[[Sequence[AgentMessage]], str | None] | None,
        post_selector_func: Callable[[Sequence[AgentMessage], str], str | None] | None,
    ) -> None:
        super().__init__(
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_descriptions,
            termination_condition,
            max_turns,
            model_client = model_client,
            selector_prompt = selector_prompt,
            allow_repeated_speaker = allow_repeated_speaker,
            selector_func = selector_func,
        )
        self._post_selector_func = post_selector_func

    async def select_speaker(self, messages: Sequence[AgentMessage]) -> str:
        speaker = await super().select_speaker(messages)
        if self._post_selector_func:
            override_speaker = self._post_selector_func(messages, speaker)
            if override_speaker:
                return override_speaker
        return speaker

class CustomGroupChat(BaseGroupChat):
    """A group chat manager that selects the next speaker using a ChatCompletion
    model and a custom selector function."""

    # TypeError: CustomGroupChatManager.__init__() missing 5 required positional arguments:
    # 'output_topic_type', 'participant_topic_types', 'participant_descriptions', 
    # 'max_turns', and 'selector_prompt'

    def __init__(
        self,
        participants: List[ChatAgent],
        model_client: ChatCompletionClient,
        *,
        termination_condition: TerminationCondition | None = None,
        max_turns: int | None = None,
        selector_prompt: str = """You are in a role play game. The following roles are available:
{roles}.
Read the following conversation. Then select the next role from {participants} to play. Only return the role.

{history}

Read the above conversation. Then select the next role from {participants} to play. Only return the role.
""",
        allow_repeated_speaker: bool = False,
        selector_func: Callable[[Sequence[AgentMessage]], str | None] | None = None,
        post_selector_func: Callable[[Sequence[AgentMessage], str], str | None] | None = None,
    ):
        super().__init__(
            participants,
            group_chat_manager_class=CustomGroupChatManager,
            termination_condition=termination_condition,
            max_turns=max_turns,
        )
        # Validate the participants.
        if len(participants) < 2:
            raise ValueError("At least two participants are required for SelectorGroupChat.")
        # Validate the selector prompt.
        if "{roles}" not in selector_prompt:
            raise ValueError("The selector prompt must contain '{roles}'")
        if "{participants}" not in selector_prompt:
            raise ValueError("The selector prompt must contain '{participants}'")
        if "{history}" not in selector_prompt:
            raise ValueError("The selector prompt must contain '{history}'")
        self._selector_prompt = selector_prompt
        self._model_client = model_client
        self._allow_repeated_speaker = allow_repeated_speaker
        self._selector_func = selector_func

        self._post_selector_func = post_selector_func

    def _create_group_chat_manager_factory(
        self,
        group_topic_type: str,
        output_topic_type: str,
        participant_topic_types: List[str],
        participant_descriptions: List[str],
        termination_condition: TerminationCondition | None,
        max_turns: int | None,
    ) -> Callable[[], BaseGroupChatManager]:
        return lambda: CustomGroupChatManager(
            group_topic_type,
            output_topic_type,
            participant_topic_types,
            participant_descriptions,
            termination_condition,
            max_turns,
            self._model_client,
            self._selector_prompt,
            self._allow_repeated_speaker,
            self._selector_func,
            self._post_selector_func,
        )

