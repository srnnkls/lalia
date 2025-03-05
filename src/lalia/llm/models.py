from enum import StrEnum

from lalia.io.logging import get_logger

MINIMUM_CONTEXT_WINDOW = 32000

logger = get_logger(__name__)


class ChatModel(StrEnum):
    GPT_4 = "gpt-4"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"
    O1_PREVIEW = "o1-preview"
    O1_MINI = "o1-mini"
    O1 = "o1"
    O1_PRO = "o1-pro"
    O3_MINI = "o3-mini"
    O3_MINI_2025_01_31 = "o3-mini-2025-01-31"

    @property
    def context_window(self) -> int:
        if context_window := CONTEXT_WINDOWS.get(self):
            return context_window
        logger.error(f"No context window defined for model {self}.")
        return MINIMUM_CONTEXT_WINDOW

    @property
    def token_limit(self) -> int:
        return self.context_window


CONTEXT_WINDOWS = {
    ChatModel.GPT_4: 8192,
    ChatModel.GPT_4_0613: 8192,
    ChatModel.GPT_4_TURBO: 32000,
    ChatModel.GPT_4_0125_PREVIEW: 32000,
    ChatModel.GPT_4_1106_PREVIEW: 32000,
    ChatModel.GPT_4O: 32000,
    ChatModel.GPT_4O_2024_05_13: 32000,
    ChatModel.GPT_4O_2024_08_06: 32000,
    ChatModel.O1_PREVIEW: 128000,
    ChatModel.O1_MINI: 128000,
    ChatModel.O1: 200000,
    ChatModel.O1_PRO: 200000,
    ChatModel.O3_MINI: 200000,
}
"""
The maximum number of tokens that can be used in a single request for each model.

We limit the large context models to 32k tokens to avoid too severe performance
degradations.

See e.g.:
- https://x.com/gregkamradt/status/1722386725635580292?t=4GUMVT7i-TW7STQKHwW33A
- https://x.com/LouisKnightWebb/status/1790265899255017893
- https://arxiv.org/pdf/2404.06654
"""
