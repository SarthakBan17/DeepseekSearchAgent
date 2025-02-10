import os
import operator
import json
from dotenv import load_dotenv, find_dotenv
from langchain_ollama import ChatOllama
from tavily import TavilyClient
from formatingStuff import deduplicate_and_format_sources, format_sources
from prompts import query_writer_instructions, summarizer_instructions
from dataclasses import dataclass, field
from typing_extensions import Annotated, TypedDict, Literal
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from configuration import Configuration


# Load the API key from the .env file
load_dotenv(find_dotenv(), override=True)  # This will load variables from .env into the environment
api_key = os.environ.get("TAVILY_API_KEY")

# Setting up the LLM
model_name = "deepseek-r1"
llm = ChatOllama(model=model_name)
llm_json = ChatOllama(model=model_name, format="json")

# Create a Client object
tavily_client = TavilyClient(api_key=api_key)

def tavily_search(query, include_raw_content=True, max_results=1):
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=include_raw_content)

# Different states that are part of the agent chain
@dataclass(kw_only=True)
class SummaryState:
    research_topic: str = field(default=None) # Report topic     
    search_query: str = field(default=None) # Search query
    web_research_results: Annotated[list, operator.add] = field(default_factory=list) 
    sources_gathered: Annotated[list, operator.add] = field(default_factory=list)
    running_summary: str = field(default=None) # Final report

@dataclass(kw_only=True)
class SummaryStateInput(TypedDict):
    research_topic: str = field(default=None) # Report topic     

@dataclass(kw_only=True)
class SummaryStateOutput(TypedDict):
    running_summary: str = field(default=None) # Final report

# Creating a generate_query function
def generate_query(state: SummaryState):
    #format the prompt with the research topic from user
    query_writer_instructions_formated = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate query
    result = llm_json.invoke(
        [SystemMessage(content=query_writer_instructions_formated),
        HumanMessage(content=f"Generate query for web search")]
    )
    query = json.loads(result.content)

    return {"search_query": query["query"]}

# Creating web search function
def web_search(state: SummaryState):
    # First perform tavily search
    search_result = tavily_search(state.search_query, include_raw_content=True, max_results=1)

    # Format the sources
    search_sources = deduplicate_and_format_sources(search_result, max_tokens_per_source=1000)

    return {"source_gathered": [format_sources(search_result)], "web_research_results": [search_sources]}

# Perform Summaery
def summarize_souces(state: SummaryState):
    # Get the most recent search result
    recent_search_result = state.web_research_results[-1]

    # Creating human message for LLM
    human_message_content = (
        f"Summarize the following web results: {recent_search_result}"
        f"That address the following topic: {state.research_topic}"
    )

    # Generate summary
    result_summary = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result_summary.content
    return {"running_summary": running_summary}

# Finalize the summary
def finalize_summary(state: SummaryState):
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)

    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}


# Adding all the functions to nodes and making a graph
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_search", web_search)
builder.add_node("summarize_souces", summarize_souces)
builder.add_node("finalize_summary", finalize_summary)

# Connecting the graph
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_search")
builder.add_edge("web_search", "summarize_souces")
builder.add_edge("summarize_souces", "finalize_summary")
builder.add_edge("finalize_summary", END)

graph = builder.compile()


# Test
research_input = SummaryStateInput(
    research_topic="What is deepseek?"
)
summary = graph.invoke(research_input)

print(summary["running_summary"])