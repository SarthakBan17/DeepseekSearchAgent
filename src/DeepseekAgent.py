import os
import json
from typing_extensions import Literal

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from utils import tavily_search
from formatingStuff import deduplicate_and_format_sources, format_sources
from prompts import query_writer_instructions, summarizer_instructions, reflection_instructions
from state import SummaryState, SummaryStateInput, SummaryStateOutput
from configuration import Configuration

# Setting up the LLM
model_name = "deepseek-r1"
llm = ChatOllama(model=model_name)
llm_json = ChatOllama(model=model_name, format="json")
llama_jason = ChatOllama(model="llama3.2", format="json")

# Creating a generate_query function
def generate_query(state: SummaryState):
    #format the prompt with the research topic from user
    query_writer_instructions_formated = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate query
    result = llama_jason.invoke(
        [SystemMessage(content=query_writer_instructions_formated),
        HumanMessage(content=f"Generate query for web search")]
    )
    query = json.loads(result.content)
    print(query)

    return {"search_query": query["query"]}

# Creating web search function
def web_search(state: SummaryState):
    # First perform tavily search
    search_result = tavily_search(state.search_query, max_results=1)

    # Format the sources
    search_sources = deduplicate_and_format_sources(search_result, max_tokens_per_source=1000)

    return {"source_gathered": [format_sources(search_result)], "research_loop_count": state.research_loop_count+1 ,"web_research_results": [search_sources]}

# Perform Summaery
def summarize_souces(state: SummaryState):

    # Existing summary 
    existing_summary = state.running_summary

    # Get the most recent search result
    recent_search_result = state.web_research_results[-1]

    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {recent_search_result} \n <New Search Results>"
        )
    else:
        # Creating human message for LLM
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {recent_search_result} \n <Search Results>"
        )   

    # Generate summary
    result_summary = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    running_summary = result_summary.content

    # It appears very challenging to prompt with the think tags, so removing them
    while "<think>" in running_summary and "</think>" in running_summary:
        start = running_summary.find("<think>")
        end = running_summary.find("</think>") + len("</think>")
        running_summary = running_summary[:start] + running_summary[end:]
    
    return {"running_summary": running_summary}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    result = llama_jason.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")]
    )
    follow_up_query = json.loads(result.content)

    # Get the follow-up query
    query = follow_up_query.get('follow_up_query')
    print(f"loop number {state.research_loop_count}: query: {query}")

    # JSON mode can fail in some cases
    if not query:

        # Fallback to a placeholder query
        return {"search_query": f"Tell me more about {state.research_topic}"}

    # Update search query with follow-up query
    return {"search_query": follow_up_query['follow_up_query']}

# Finalize the summary
def finalize_summary(state: SummaryState):
    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)

    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_search"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_search_loops:
        return "web_search"
    else:
        return "finalize_summary"

# Adding all the functions to nodes and making a graph
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_search", web_search)
builder.add_node("summarize_souces", summarize_souces)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Connecting the graph
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_search")
builder.add_edge("web_search", "summarize_souces")
builder.add_edge("summarize_souces", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()


# Test
research_input = SummaryStateInput(
    research_topic="Analyse Meta Company"
)
summary = graph.invoke(research_input)

print(summary["running_summary"])