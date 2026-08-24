from dataclasses import dataclass
from typing import Callable, List, Optional, Union

from pydantic import BaseModel


class MalformedToolCallError(ValueError):
    """Model output used tool-like syntax that the configured parser cannot trust."""


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []
    # Some model protocols can emit ordinary text both before and after tool
    # calls in one decoded increment. ``normal_text`` plus ``calls`` cannot
    # preserve that ordering, so detectors may provide the exact sequence here.
    # Callers that do not understand it can keep using the legacy fields.
    ordered_items: Optional[List[Union[str, ToolCallItem]]] = None


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""
_GetInfoFunc = Callable[[str], StructureInfo]
