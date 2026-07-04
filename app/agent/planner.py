"""
Core orchestration: decides whether a user message needs knowledge-base
retrieval, a mock tool call, or a direct response, executes accordingly,
and produces the final natural-language reply.

Pattern used: single-agent-with-tools. One OpenAI chat completion call is
given all three callable "tools" (get_order_status, search_product,
search_knowledge_base) plus the conversation history. The model decides
which (if any) to call. We execute the call(s), feed the results back in
a second completion call, and return that as the final answer. This
keeps routing logic inside the LLM's native function-calling instead of
a hand-rolled intent classifier, which is more robust to phrasing
variation ("where's my package ORD001" vs "status of ORD001").
"""
import json
from typing import List

from openai import OpenAI

from app.config import CHAT_MODEL, OPENAI_API_KEY, NO_ANSWER_MESSAGE
from app.agent.prompts import SYSTEM_PROMPT, KB_ANSWER_INSTRUCTIONS
from app.memory import session_memory
from app.models.schemas import ChatResponse, SourceChunk, ToolCallTrace
from app.tools.order_status import get_order_status, TOOL_SCHEMA as ORDER_TOOL_SCHEMA
from app.tools.product_search import search_product, TOOL_SCHEMA as PRODUCT_TOOL_SCHEMA
from app.retrieval.retriever import retrieve, format_context, TOOL_SCHEMA as KB_TOOL_SCHEMA

_client = OpenAI(api_key=OPENAI_API_KEY)

TOOLS = [ORDER_TOOL_SCHEMA, PRODUCT_TOOL_SCHEMA, KB_TOOL_SCHEMA]


def _execute_tool_call(name: str, arguments: dict) -> tuple[dict, List[SourceChunk]]:
    """Executes one tool call by name. Returns (result_dict, source_chunks)."""
    sources: List[SourceChunk] = []

    if name == "get_order_status":
        result = get_order_status(arguments.get("order_id", ""))

    elif name == "search_product":
        result = search_product(arguments.get("product_name", ""))

    elif name == "search_knowledge_base":
        query = arguments.get("query", "")
        chunks = retrieve(query)
        if not chunks:
            result = {"context": "", "found": False, "message": NO_ANSWER_MESSAGE}
        else:
            sources = [
                SourceChunk(source=c.source, chunk_id=c.chunk_id, snippet=c.text[:200])
                for c in chunks
            ]
            result = {"context": format_context(chunks), "found": True}

    else:
        result = {"error": f"Unknown tool '{name}'"}

    return result, sources


def handle_chat(session_id: str, message: str) -> ChatResponse:
    # 1. Load memory
    history = session_memory.get_history(session_id)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    # 2. First completion: let the model decide whether to call a tool
    first_response = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    choice = first_response.choices[0].message

    tool_calls_trace: List[ToolCallTrace] = []
    all_sources: List[SourceChunk] = []
    route = "direct"

    if choice.tool_calls:
        # 3. Execute every requested tool call
        messages.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [tc.model_dump() for tc in choice.tool_calls],
            }
        )

        for tool_call in choice.tool_calls:
            name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = {}

            result, sources = _execute_tool_call(name, arguments)
            all_sources.extend(sources)
            tool_calls_trace.append(
                ToolCallTrace(tool_name=name, tool_input=arguments, tool_output=result)
            )
            route = "knowledge_base" if name == "search_knowledge_base" else "tool_call"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

        # 4. Second completion: produce the final grounded answer
        final_response = _client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
        )
        reply = final_response.choices[0].message.content or NO_ANSWER_MESSAGE

    else:
        reply = choice.content or ""

    # 5. Persist memory
    session_memory.add_turn(session_id, "user", message)
    session_memory.add_turn(session_id, "assistant", reply)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        route=route,
        sources=all_sources,
        tool_calls=tool_calls_trace,
    )
