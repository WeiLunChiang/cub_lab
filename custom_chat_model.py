# %%
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

# %%

from typing import Any, Dict, List, Optional, Iterator, Callable
import openai
from pydantic import BaseModel, Field
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatResult, ChatGenerationChunk
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_openai import ChatOpenAI


# %%
class CustomChatModelAdvanced(BaseChatModel):
    api_key: str = Field(..., description="The API key for OpenAI")
    model_name: str = Field(default="gpt-4", description="The name of the model")
    n: int = Field(
        default=5,
        description="The number of characters from the last message of the prompt to be echoed",
    )
    model: Optional[ChatOpenAI] = Field(
        default=None, description="The chat model instance"
    )
    model_base: Optional[Callable] = Field(
        default=None, description="The chat model initfunction"
    )
    chat_model: Optional[ChatOpenAI] = Field(
        default=None, description="The chat model instance"
    )

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.chat_model = self.model_base(
            api_key=self.api_key, model_name=self.model_name
        )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """這段要改成打護欄API並收response的邏輯"""
        openai.api_key = self.api_key

        message = self.chat_model.invoke(messages)

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream the output of the model.

        This method generates output in a streaming fashion.
        """
        openai.api_key = self.api_key

        # Assume the model has a stream_invoke method that supports streaming
        for token in self.chat_model.stream_invoke(messages, stop=stop):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=chunk)
            yield chunk

        # Add some other information (e.g., response metadata) at the end of streaming
        chunk = ChatGenerationChunk(
            message=AIMessageChunk(content="", response_metadata={"time_in_sec": 3})
        )
        if run_manager:
            run_manager.on_llm_new_token("", chunk=chunk)
        yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "echoing-chat-model-advanced"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            "model_name": self.model_name,
        }


# %%

import os

model = CustomChatModelAdvanced(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="gpt-4o",
    n=5,
    model_base=ChatOpenAI,
)

# %%
model.invoke(
    [
        HumanMessage(content="hello!, my name is Wei"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="what is am i?"),
    ]
)
# %%

model = ChatOpenAI()
# %%
model.invoke(
    [
        HumanMessage(content="hello!"),
        AIMessage(content="Hi there human!"),
        HumanMessage(content="Meow!"),
    ]
)
# %%
