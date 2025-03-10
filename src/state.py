import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Literal


# Different states that are part of the agent chain
@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic     
    search_query: str = field(default=None) # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    research_loop_count: int = field(default=0) # Research loop count
    running_summary: str = field(default=None) # Final report

@dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    running_summary: str = field(default=None) # Final report