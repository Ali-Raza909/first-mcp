from dotenv import load_dotenv
from groq import AsyncGroq
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from pydantic import AnyUrl

from typing import List, Dict, Tuple, TypedDict
from contextlib import AsyncExitStack

import json
import asyncio
import os
import re
import shlex


load_dotenv()


class ToolDefinition(TypedDict):
    type: str
    function: dict


def _get(obj, *names, default=None):
    """Read the first attribute that exists (handles mcp 1.x camelCase
    and mcp 2.x snake_case names)."""
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


class MCP_ChatBot:

    def __init__(self):
        # MCP sessions
        self.sessions: List[ClientSession] = []
        self.exit_stack = AsyncExitStack()

        # Groq client
        self.groq = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "openai/gpt-oss-120b"

        # Tools (Groq/OpenAI format) and which session owns each
        self.available_tools: List[ToolDefinition] = []
        self.tool_to_session: Dict[str, ClientSession] = {}

        # Resources: static URIs and URI templates, each mapped to a session
        self.resource_to_session: Dict[str, ClientSession] = {}
        self.resource_templates: List[Tuple[str, ClientSession]] = []

        # Prompts: name -> (prompt object, session)
        self.prompts: Dict[str, Tuple[object, ClientSession]] = {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server.

        Each server gets its own exit stack, so a server that fails to start
        is cleaned up immediately and can't break shutdown for the others.
        """
        local_stack = AsyncExitStack()

        try:
            if "url" in server_config:
                # Remote server over SSE, e.g. {"url": "http://127.0.0.1:8001/sse"}
                transport = sse_client(
                    server_config["url"],
                    headers=server_config.get("headers"),
                    # Free Render instances sleep and can take ~30-60s to wake
                    timeout=server_config.get("timeout", 60),
                )
            else:
                # Local server launched as a subprocess over stdio
                transport = stdio_client(StdioServerParameters(**server_config))

            read, write = await local_stack.enter_async_context(transport)

            session = await local_stack.enter_async_context(
                ClientSession(read, write)
            )

            await session.initialize()

            # ---- Tools ----
            response = await session.list_tools()
            tools = response.tools

            new_tools = []
            for tool in tools:
                schema = _get(
                    tool, "input_schema", "inputSchema",
                    default={"type": "object", "properties": {}},
                )
                new_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description or "",
                        "parameters": schema,
                    },
                })

            # ---- Resources (optional: not every server supports them) ----
            static_uris: List[str] = []
            template_uris: List[str] = []

            try:
                res = await session.list_resources()
                static_uris = [str(r.uri).rstrip("/") for r in res.resources]
            except Exception:
                pass

            try:
                tmpl = await session.list_resource_templates()
                templates = _get(
                    tmpl, "resource_templates", "resourceTemplates", default=[]
                )
                template_uris = [
                    str(_get(t, "uri_template", "uriTemplate")) for t in templates
                ]
            except Exception:
                pass

            # ---- Prompts (optional) ----
            prompt_list = []
            try:
                pr = await session.list_prompts()
                prompt_list = pr.prompts
            except Exception:
                pass

            print(f"\nConnected to {server_name}:")
            print(f"  tools:     {[t.name for t in tools]}")
            if static_uris or template_uris:
                print(f"  resources: {static_uris + template_uris}")
            if prompt_list:
                print(f"  prompts:   {[p.name for p in prompt_list]}")

            # Only register after everything succeeded
            for tool, definition in zip(tools, new_tools):
                self.tool_to_session[tool.name] = session
                self.available_tools.append(definition)

            for uri in static_uris:
                self.resource_to_session[uri] = session

            for template in template_uris:
                self.resource_templates.append((template, session))

            for prompt in prompt_list:
                self.prompts[prompt.name] = (prompt, session)

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

    # ------------------------------------------------------------------
    # Resources
    # ------------------------------------------------------------------

    def _find_resource_session(self, uri: str):
        """Find which session serves a URI (static first, then templates)."""
        normalized = uri.rstrip("/")

        if normalized in self.resource_to_session:
            return self.resource_to_session[normalized]

        for template, session in self.resource_templates:
            # Turn "papers://{topic}" into a regex like "^papers://[^/]+$"
            parts = re.split(r"\{[^}]+\}", template)
            pattern = "^" + "[^/]+".join(re.escape(p) for p in parts) + "$"
            if re.match(pattern, normalized):
                return session

        return None

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI and return its text."""
        session = self._find_resource_session(uri)
        if session is None:
            return f"Error: no connected server provides resource {uri}"

        try:
            result = await session.read_resource(AnyUrl(uri))
        except Exception as e:
            return f"Error reading {uri}: {e}"

        parts = []
        for item in result.contents:
            text = getattr(item, "text", None)
            parts.append(text if text is not None else str(item))
        return "\n".join(parts)

    async def handle_resource_command(self, query: str):
        """Handle `@folders` or `@topic [optional question]`."""
        token, _, question = query[1:].partition(" ")
        question = question.strip()

        if not token:
            print("Usage: @folders  or  @<topic> [question]")
            return

        uri = f"papers://{token}"
        content = await self.read_resource(uri)

        if not question:
            print(f"\n{content}")
            return

        # Question after the resource: give the model the resource as context
        await self.process_query(
            f"Use this resource as context:\n\n{content}\n\nQuestion: {question}"
        )

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def list_prompts(self):
        if not self.prompts:
            print("\nNo prompts available.")
            return

        print("\nAvailable prompts:")
        for name, (prompt, _) in self.prompts.items():
            print(f"  - {name}: {prompt.description or ''}")
            for arg in prompt.arguments or []:
                required = " (required)" if arg.required else ""
                print(f"      {arg.name}{required}")

        print('\nUsage: /prompt <name> key=value key2="value with spaces"')

    async def run_prompt(self, args: List[str]):
        """Handle `/prompt <name> key=value ...`."""
        if not args:
            self.list_prompts()
            return

        name = args[0]
        if name not in self.prompts:
            print(f"\nUnknown prompt: {name}")
            self.list_prompts()
            return

        _, session = self.prompts[name]

        arguments = {}
        for pair in args[1:]:
            if "=" not in pair:
                print(f"\nIgnoring '{pair}' (expected key=value)")
                continue
            key, value = pair.split("=", 1)
            arguments[key] = value

        try:
            result = await session.get_prompt(name, arguments=arguments)
        except Exception as e:
            print(f"\nCould not load prompt '{name}': {e}")
            return

        texts = []
        for message in result.messages:
            content = message.content
            text = getattr(content, "text", None)
            texts.append(text if text is not None else str(content))

        prompt_text = "\n".join(texts)
        print(f"\nRunning prompt '{name}'...")
        await self.process_query(prompt_text)

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

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

                        if _get(result, "is_error", "isError", default=False):
                            tool_output = "Tool execution error: " + tool_output

                    except Exception as e:
                        tool_output = f"Tool execution failed: {str(e)}"

                print(f"Tool result: {tool_output[:500]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output,
                })

    def print_help(self):
        print("\nCommands:")
        print("  @folders                 list saved topics (resource)")
        print("  @<topic>                 show papers saved for a topic (resource)")
        print("  @<topic> <question>      ask a question using that topic as context")
        print("  /prompts                 list available prompts")
        print("  /prompt <name> k=v ...   run a prompt, e.g.")
        print('                           /prompt generate_search_prompt topic="machine learning" num_papers=3')
        print("  /help                    show this help")
        print("  quit                     exit")

    async def chat_loop(self):
        """Run an interactive chat loop."""

        print("\nMCP Chatbot Started!")
        print("Powered by Groq + MCP")
        print("Type a query, use @resource / /prompt commands, /help, or 'quit'.")

        while True:
            try:
                query = await asyncio.to_thread(input, "\nQuery: ")
                query = query.strip()

                if not query:
                    continue

                if query.lower() == "quit":
                    break

                if query.startswith("@"):
                    await self.handle_resource_command(query)
                    continue

                if query.startswith("/"):
                    try:
                        parts = shlex.split(query)
                    except ValueError as e:
                        print(f"\nCould not parse command: {e}")
                        continue

                    command = parts[0].lower()

                    if command == "/prompts":
                        self.list_prompts()
                    elif command == "/prompt":
                        await self.run_prompt(parts[1:])
                    elif command == "/help":
                        self.print_help()
                    else:
                        print(f"\nUnknown command: {command}")
                        self.print_help()
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