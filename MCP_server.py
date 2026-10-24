import arxiv  #for searching papers
import json   #for some formatting
import os  # for env vars
import requests
from typing import List
from dotenv import load_dotenv
from groq import Groq
PAPER_DIR = "papers"  # for saving info to the file system


class _TimeoutSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (5, 15))
        return super().request(method, url, **kwargs)


def search_papers(topic: str, max_results: int = 5) -> List[str]:
    client = arxiv.Client(num_retries=0)
    client._session = _TimeoutSession()

    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)

    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)

    file_path = os.path.join(path, "papers_info.json")

    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info

    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)

    print(f"Results are saved in: {file_path}")

    return paper_ids

# func to search for info about a specific paper

def extract_info(paper_id: str) -> str:
  
 
    for item in os.listdir(PAPER_DIR):
        item_path = os.path.join(PAPER_DIR, item)
        if os.path.isdir(item_path):
            file_path = os.path.join(item_path, "papers_info.json")
            if os.path.isfile(file_path):
                try:
                    with open(file_path, "r") as json_file:
                        papers_info = json.load(json_file)
                        if paper_id in papers_info:
                            return json.dumps(papers_info[paper_id], indent=2)
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error reading {file_path}: {str(e)}")
                    continue
    
    return f"There's no saved information related to paper {paper_id}."

# tools for searching paper and extracting info bout em


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_papers",
            "description": (
                "Search arXiv papers by topic and save "
                "their information locally."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Topic to search for"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum papers to retrieve"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "extract_info",
            "description": (
                "Retrieve saved information about an arXiv paper."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_id": {
                        "type": "string",
                        "description": "The arXiv paper ID"
                    }
                },
                "required": ["paper_id"]
            }
        }
    }
]


# func to maps our tools to their underline funs 

mapping_tool_function = {
    "search_papers": search_papers,
    "extract_info": extract_info
}

def execute_tool(tool_name, tool_args):
    
    result = mapping_tool_function[tool_name](**tool_args)

    if result is None:
        result = "The operation completed but didn't return any results."
        
    elif isinstance(result, list):
        result = ', '.join(result)
        
    elif isinstance(result, dict):
        # convert dictionaries to formatted JSON strings
        result = json.dumps(result, indent=2)
    
    else:
        # for any other type, convert using str()
        result = str(result)
    return result


load_dotenv() # for env vars
client = Groq()

def process_query(query):
    messages = [
        {"role": "user", "content": query}
    ]

    # Prevent an endless tool-calling loop
    for _ in range(10):
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_completion_tokens=2024
        )

        assistant_message = response.choices[0].message

        # Preserve the assistant's tool calls in the conversation
        messages.append(
            assistant_message.model_dump(exclude_none=True)
        )

        # If no tools were requested, display the final answer
        if not assistant_message.tool_calls:
            if assistant_message.content:
                print(assistant_message.content)
            return

        # Execute each requested tool and return its result
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name

            try:
                tool_args = json.loads(
                    tool_call.function.arguments
                )

                print(
                    f"Calling tool {tool_name} "
                    f"with args {tool_args}"
                )

                if tool_name not in mapping_tool_function:
                    raise ValueError(
                        f"Unknown tool: {tool_name}"
                    )

                result = execute_tool(tool_name, tool_args)

            except Exception as e:
                result = f"Tool execution error: {e}"

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    print("Stopped: maximum tool-calling iterations reached.")





def chat_loop():
    print("Type your queries or 'quit' to exit.")
    while True:
        try:
            query = input("\nQuery: ").strip()
            if query.lower() == 'quit':
                break
    
            process_query(query)
            print("\n")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    chat_loop()