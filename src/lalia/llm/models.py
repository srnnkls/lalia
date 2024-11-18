from enum import StrEnum

MINIMUM_CONTEXT_WINDOW = 4096


class ChatModel(StrEnum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_3_5_TURBO_0613 = "gpt-3.5-turbo-0613"
    GPT_3_5_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT_3_5_TURBO_0125 = "gpt-3.5-turbo-0125"
    GPT_4 = "gpt-4"
    GPT_4_0613 = "gpt-4-0613"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4_0125_PREVIEW = "gpt-4-0125-preview"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_2024_08_06 = "gpt-4o-2024-08-06"

    @property
    def context_window(self) -> int:
        return CONTEXT_WINDOWS.get(self, MINIMUM_CONTEXT_WINDOW)

    @property
    def token_limit(self) -> int:
        return self.context_window


CONTEXT_WINDOWS = {
    ChatModel.GPT_3_5_TURBO: 4096,
    ChatModel.GPT_3_5_TURBO_0613: 4096,
    ChatModel.GPT_3_5_TURBO_1106: 4096,
    ChatModel.GPT_3_5_TURBO_0125: 4096,
    ChatModel.GPT_4: 8192,
    ChatModel.GPT_4_0613: 8192,
    ChatModel.GPT_4_TURBO: 32000,
    ChatModel.GPT_4_0125_PREVIEW: 32000,
    ChatModel.GPT_4_1106_PREVIEW: 32000,
    ChatModel.GPT_4O: 32000,
    ChatModel.GPT_4O_2024_05_13: 32000,
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


class FunctionCallDirective(StrEnum):
    NONE = "none"
    AUTO = "auto"
