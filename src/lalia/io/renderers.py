from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

from rich.console import Group
from rich.json import JSON
from rich.jupyter import JupyterMixin
from rich.panel import Panel
from rich.table import Table, box
from rich.text import Text

from lalia.chat import messages
from lalia.chat.messages import tags
from lalia.chat.roles import Role

TEXT_COLOR = "grey89"


class TagColor(StrEnum):
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    BLUE = "blue"
    MAGENTA = "magenta"
    CYAN = "cyan"
    BLUE3 = "blue3"
    TURQUOISE4 = "turquoise4"


"""
Mapping of tag colors to styles.
"""
TAG_STYLES = {color: f"{TEXT_COLOR} on {color}" for color in TagColor}


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
            if message.tags:
                tags = " ".join(
                    TagRenderer(tag).__rich__().markup for tag in message.tags
                )
            else:
                tags = ""

            content = (
                JSON(json.dumps(asdict(message.function_call)))
                if isinstance(message, messages.AssistantMessage)
                and message.function_call
                else message.content
            )
            row = self._format_row(timestamp, tags, role, content)
            table.add_row(*row)
        return table

    def _format_row(
        self, timestamp: datetime, tags: str, role: Role, content: str | JSON | None
    ) -> tuple[str, str | Group] | tuple[str, str, str | Group]:
        role_formatted = f"[{role.color}]{role}[/{role.color}]"

        if content is None:
            content = "null"
        if isinstance(content, str) and len(content) > self.max_cell_length:
            content = f"{content[:self.max_cell_length-4]}..."
        if isinstance(content, JSON):
            content_formatted = (
                Group(tags, Text(""), content) if tags else Group(content)
            )
        else:
            content_formatted = (
                f"{tags}\n\n[{role.color}]{content}[/{role.color}]"
                if tags
                else f"[{role.color}]{content}[/{role.color}]"
            )

        if self.include_timestamps:
            timetamp_formatted = (
                f"[{role.color}]{timestamp:%y-%m-%d}\n"
                f"{timestamp:%H:%M:%S}[/{role.color}]"
            )
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


class TagRenderer(JupyterMixin):
    colors: ClassVar[deque[TagColor]] = deque(TagColor)
    key_registry: ClassVar[dict[str, TagColor]] = {}

    def __init__(self, tag: tags.Tag):
        self.tag = tag

    def __rich__(self) -> Text:
        if self.tag.color is None:
            color = self.get_color(self.tag.key)
        else:
            color = self.tag.color

        return Text.from_markup(
            f"[b]{self.tag.key}:[/b] {self.tag.value}",
            style=TAG_STYLES[color],
        )

    @classmethod
    def get_color(cls, key: str) -> TagColor:
        if key in cls.key_registry and tags.Tag.group_colors_by_key:
            return cls.key_registry[key]

        if not cls.colors:
            cls.colors = deque(TagColor)

        return cls.colors.popleft()

    @classmethod
    def register_key(cls, key: str, color: TagColor):
        if key not in cls.key_registry:
            cls.key_registry[key] = color
            if color in cls.colors:
                cls.colors.remove(color)
