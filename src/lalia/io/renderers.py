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

from lalia.chat.messages import messages, tags
from lalia.chat.messages.fold_state import FoldState
from lalia.chat.roles import Role

TEXT_COLOR = "grey15"


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

"""
Mapping of tag colors to styles for folded tags.
"""
FOLDED_TAG_STYLES = {color: f"{color} dim" for color in TagColor}

ROW_STYLE = ""
FOLDED_ROW_STYLE = "dim"


class ConversationRenderer(JupyterMixin):
    max_cell_length = 2000
    cell_width = 80
    include_timestamps = True

    def __init__(
        self,
        messages: Sequence[messages.Message],
        fold_states: Sequence[FoldState],
        title: str | None = None,
    ):
        self.messages = messages
        self.fold_states = fold_states
        self.title = title

    def __rich__(self) -> Table:
        table = Table(
            title=self.title,
            show_header=False,
            box=box.SIMPLE,
            width=self.cell_width,
            padding=(1, 0, 0, 0),
        )
        if self.include_timestamps:
            table.add_column("Timestamp")
        table.add_column("Role")
        table.add_column("Message")

        for message, fold in zip(self.messages, self.fold_states, strict=True):
            timestamp = message.timestamp
            role = message.to_base_message().role

            content = (
                JSON(
                    json.dumps(
                        {
                            "name": message.function_call.name,
                            "arguments": message.function_call.arguments,
                        }
                    )
                )
                if isinstance(message, messages.AssistantMessage)
                and message.function_call
                else message.content
            )
            row = self._format_row(timestamp, message.tags, role, content, fold)

            row_style = ROW_STYLE if fold is FoldState.UNFOLDED else FOLDED_ROW_STYLE
            table.add_row(
                *row,
                style=f"{role.color} {row_style}",
            )
        return table

    def _format_content(
        self, content: str | JSON | None, fold_state: FoldState
    ) -> Text | JSON:
        if fold_state is FoldState.FOLDED:
            content_formatted = Text(f"--- folded ({len(str(content))} characters) ---")
        elif content is None:
            content_formatted = Text("null")
        elif isinstance(content, str) and len(content) > self.max_cell_length:
            content_formatted = Text(f"{content[: self.max_cell_length]} ...")
        elif isinstance(content, JSON):
            content_formatted = content
        else:
            try:
                content_formatted = JSON(content)
            except json.JSONDecodeError:
                content_formatted = Text(content)
        return content_formatted

    def _format_row(
        self,
        timestamp: datetime,
        tags: set[tags.Tag],
        role: Role,
        content: str | JSON | None,
        fold_state: FoldState,
    ) -> tuple[Text, Group] | tuple[Text, Text, Group]:
        role_formatted = Text(role)
        content_formatted = Group(self._format_content(content, fold_state))
        if tags:
            tags_formatted = Text.from_markup(
                " ".join(
                    [
                        TagRenderer(tag, fold_state).__rich__().markup
                        for tag in sorted(tags, key=lambda tag: tag.key)
                    ]
                )
            )
            content_formatted = Group(tags_formatted, content_formatted)

        if self.include_timestamps:
            timestamp_formatted = Text(f"{timestamp:%y-%m-%d}\n{timestamp:%H:%M:%S}")
            return timestamp_formatted, role_formatted, content_formatted
        else:
            return role_formatted, content_formatted


class MessageRenderer(JupyterMixin):
    def __init__(self, message: messages.Message):
        self.message = message

    def __rich__(self) -> Table:
        return ConversationRenderer([self.message], [FoldState.UNFOLDED]).__rich__()


class MessageBufferRender(JupyterMixin):
    panel_width = 88

    def __init__(
        self,
        messages: Sequence[messages.Message],
        pending: Sequence[messages.Message],
        message_fold_states: Sequence[FoldState],
        pending_fold_states: Sequence[FoldState],
    ):
        self.messages = messages
        self.pending = pending
        self.message_fold_states = message_fold_states
        self.pending_fold_states = pending_fold_states

    def __rich__(self) -> Group:
        messages = ConversationRenderer(
            self.messages, self.message_fold_states
        ).__rich__()
        if self.pending:
            pending = ConversationRenderer(
                self.pending, self.pending_fold_states
            ).__rich__()
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

    def __init__(
        self,
        tag: tags.Tag,
        fold_state: FoldState = FoldState.UNFOLDED,
    ):
        self.tag = tag
        self.fold_state = fold_state

    def __rich__(self) -> Text:
        if self.tag.color is None:
            color = self.get_color(self.tag.key)
        else:
            color = self.tag.color

        return Text.from_markup(
            f"[b] {self.tag.key}:[/b] {self.tag.value} ",
            style=TAG_STYLES[color]
            if self.fold_state is FoldState.UNFOLDED
            else FOLDED_TAG_STYLES[color],
        )
