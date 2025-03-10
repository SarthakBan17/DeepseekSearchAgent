import os

from tavily import TavilyClient

# Create a Client object
tavily_client = TavilyClient()

'''response = tavily_client.search("Who is leo messi", max_results=1)
print(response)'''

def tavily_search(query, include_raw_content=True, max_results=1):
    return tavily_client.search(query, 
                         max_results=max_results, 
                         include_raw_content=include_raw_content)