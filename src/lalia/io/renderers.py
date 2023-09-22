from __future__ import annotations

import json
from dataclasses import asdict

from rich.json import JSON
from rich.jupyter import JupyterMixin
from rich.table import Table, box

from lalia.chat import messages


class ConversationRenderer(JupyterMixin):
    max_cell_length = 2000

    def __init__(self, messages: list[messages.Message]):
        self.messages = messages

    def __rich__(self) -> Table:
        table = Table(show_header=False, box=box.SIMPLE, width=88)
        table.add_column("Role", style="dim")
        table.add_column("Message")

        for message in self.messages:
            role = message.to_base_message().role
            content = (
                JSON(json.dumps(asdict(message.function_call)))
                if isinstance(message, messages.AssistantMessage)
                and message.function_call
                else message.content
            )
            if isinstance(content, str) and len(content) > self.max_cell_length:
                content = f"{content[:self.max_cell_length-4]}..."
            table.add_row(
                f"[{role.color}]{role}[/{role.color}]",
                content
                if isinstance(content, JSON)
                else f"[{role.color}]{content}[/{role.color}]",
            )
        return table


class MessageRenderer(JupyterMixin):
    def __init__(self, message: messages.Message):
        self.message = message

    def __rich__(self) -> Table:
        return ConversationRenderer([self.message]).__rich__()
