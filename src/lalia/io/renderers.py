from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime

from rich.console import Group
from rich.json import JSON
from rich.jupyter import JupyterMixin
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text

from lalia.chat import messages
from lalia.chat.roles import Role


class ConversationRenderer(JupyterMixin):
    max_cell_length = 2000
    cell_width = 80
    include_timestamps = True

    def __init__(self, messages: Sequence[messages.Message], title: str | None = None):
        self.messages = messages
        self.title = title

    def __rich__(self) -> Table:
        table = Table(
            title=self.title, show_header=False, box=box.SIMPLE, width=self.cell_width
        )
        if self.include_timestamps:
            table.add_column("Timestamp", style="dim")
        table.add_column("Role", style="dim")
        table.add_column("Message")

        for message in self.messages:
            timestamp = message.to_base_message().timestamp
            role = message.to_base_message().role
            content = (
                JSON(json.dumps(asdict(message.function_call)))
                if isinstance(message, messages.AssistantMessage)
                and message.function_call
                else message.content
            )
            row = self._format_row(timestamp, role, content)
            table.add_row(*row)
        return table


    def _format_row(
        self, timestamp: datetime, role: Role, content: str | JSON | None
    ) -> tuple[str, str | JSON] | tuple[str, str, str | JSON]:
        role_formatted = (f"[{role.color}]{role}[/{role.color}]")

        if content is None:
            content = "null"
        if isinstance(content, str) and len(content) > self.max_cell_length:
            content = f"{content[:self.max_cell_length-4]}..."
        if isinstance(content, JSON):
            content_formatted = content
        else:
            content_formatted = f"[{role.color}]{content}[/{role.color}]"

        if self.include_timestamps:
            timetamp_formatted = (f"[{role.color}]{timestamp:%H:%M:%S}[/{role.color}]")
            return timetamp_formatted, role_formatted, content_formatted
        else:
            return role_formatted, content_formatted



class MessageRenderer(JupyterMixin):
    def __init__(self, message: messages.Message):
        self.message = message

    def __rich__(self) -> Table:
        return ConversationRenderer([self.message]).__rich__()


class MessageBufferRender(JupyterMixin):
    panel_width = 88

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
                Panel(messages, title="Messages", width=self.panel_width),
                Text("\n"),
                Panel(
                    pending, title="Pending", box=box.MINIMAL, width=self.panel_width
                ),
            )
        else:
            return Group(Panel(messages, title="Messages", width=self.panel_width))
