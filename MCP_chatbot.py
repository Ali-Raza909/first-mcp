from dotenv import load_dotenv
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack

import json
import asyncio
import os
# jiooooooooooooooo

load_dotenv()


class ToolDefinition(TypedDict):
    type: str
    function: dict


class MCP_ChatBot:

    def __init__(self):
        # MCP sessions
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()

        # Groq client
        self.groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"

        # Available MCP tools (in Groq/OpenAI format)
        self.available_tools: List[ToolDefinition] = []

        # Map tool names to their MCP sessions
        self.tool_to_session: Dict[str, ClientSession] = {}

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server.

        Each server gets its own exit stack, so a server that fails to start
        is cleaned up immediately and can't break shutdown for the others.
        """
        local_stack = AsyncExitStack()

        try:
            server_params = StdioServerParameters(**server_config)

            read, write = await local_stack.enter_async_context(
                stdio_client(server_params)
            )

            session = await local_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()

            response = await session.list_tools()
            tools = response.tools

            print(
                f"\nConnected to {server_name} with tools:",
                [tool.name for tool in tools],
            )

            new_tools = []
            for tool in tools:
                # Newer mcp versions use input_schema, older use inputSchema
                schema = (
                    getattr(tool, "input_schema", None)
                    or getattr(tool, "inputSchema", None)
                    or {"type": "object", "properties": {}}
                )

                new_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                })

            # Only register after everything succeeded
            for tool, definition in zip(tools, new_tools):
                self.tool_to_session[tool.name] = session
                self.available_tools.append(definition)

            self.sessions.append(session)

            # Hand ownership of this server's resources to the main stack
            self.exit_stack.push_async_callback(local_stack.aclose)

        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")
            try:
                await local_stack.aclose()
            except BaseException:
                pass

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)

            servers = data.get("mcpServers", {})

            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)

        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query: str):
        """Process a user query using Groq and MCP tools."""

        messages = [{"role": "user", "content": query}]

        while True:
            kwargs = {}
            if self.available_tools:
                kwargs["tools"] = self.available_tools
                kwargs["tool_choice"] = "auto"

            response = await self.groq.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=2024,
                **kwargs,
            )

            assistant_message = response.choices[0].message

            # No tool calls: print the final answer and stop
            if not assistant_message.tool_calls:
                if assistant_message.content:
                    print(assistant_message.content)
                break

            # Preserve the assistant's tool-call message
            messages.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    tool_call.model_dump(exclude_none=True)
                    for tool_call in assistant_message.tool_calls
                ],
            })

            # Execute every requested tool
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name

                try:
                    tool_args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    tool_args = {}

                print(f"\nCalling tool {tool_name} with args {tool_args}")

                session = self.tool_to_session.get(tool_name)

                if session is None:
                    tool_output = f"Error: Tool {tool_name} not found."
                else:
                    try:
                        result = await session.call_tool(
                            tool_name, arguments=tool_args
                        )

                        output_parts = []
                        for item in result.content:
                            if item.type == "text":
                                output_parts.append(item.text)
                            else:
                                output_parts.append(
                                    json.dumps(
                                        item.model_dump(mode="json"),
                                        default=str,
                                    )
                                )

                        tool_output = "\n".join(output_parts)

                        # Newer mcp: is_error, older: isError
                        if getattr(result, "is_error", False) or getattr(
                            result, "isError", False
                        ):
                            tool_output = "Tool execution error: " + tool_output

                    except Exception as e:
                        tool_output = f"Tool execution failed: {str(e)}"

                print(f"Tool result: {tool_output[:500]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                })

    async def chat_loop(self):
        """Run an interactive chat loop."""

        print("\nMCP Chatbot Started!")
        print("Powered by Groq + MCP")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = await asyncio.to_thread(input, "\nQuery: ")
                query = query.strip()

                if query.lower() == "quit":
                    break

                if not query:
                    continue

                await self.process_query(query)

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Cleanly close all MCP connections."""
        try:
            await self.exit_stack.aclose()
        except BaseException:
            # Shutdown noise (e.g. cancelled scopes after Ctrl+C) isn't useful
            pass


async def main():
    chatbot = MCP_ChatBot()

    try:
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        await chatbot.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass