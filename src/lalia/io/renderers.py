from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict

from rich.console import Group
from rich.json import JSON
from rich.jupyter import JupyterMixin
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text

from lalia.chat import messages


class ConversationRenderer(JupyterMixin):
    max_cell_length = 2000

    def __init__(self, messages: Sequence[messages.Message], title: str | None = None):
        self.messages = messages
        self.title = title

    def __rich__(self) -> Table:
        table = Table(title=self.title, show_header=False, box=box.SIMPLE, width=88)
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


class MessageBufferRender(JupyterMixin):
    def __init__(
        self, messages: Sequence[messages.Message], pending: Sequence[messages.Message]
    ):
        self.messages = messages
        self.pending = pending

    def __rich__(self) -> Group:
        messages = ConversationRenderer(self.messages).__rich__()
        if self.pending:
            pending = ConversationRenderer(self.pending).__rich__()
            return Group(
                Panel(messages, title="Messages"),
                Text("\n"),
                Panel(pending, title="Pending", box=box.MINIMAL),
            )
        else:
            return Group(Panel(messages, title="Messages"))
