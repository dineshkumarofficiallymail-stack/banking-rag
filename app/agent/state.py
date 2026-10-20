"""LangGraph state. `add_messages` is a reducer: each node returns new messages and the graph appends them."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
