from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from tkinter import END
from typing import Literal, Optional
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from ..stores import RawMessage, RawMessageStore, SemanticStore, SemanticMemory, ImageManager
from ..config import TIME_FMT, MemoryManagerConfig

class MemoryManagerTools:
    """Tools for MemoryManager to query storage layers"""
    raw: RawMessageStore
    semantic_store: SemanticStore
    image_manager: ImageManager
    tool_call_id: str
    
    max_raw_msg: int = 20
    
    def __init__(
        self,
        raw_store: RawMessageStore,
        semantic_store: SemanticStore,
        image_manager: ImageManager,
        max_raw_msg: int = 20,
    ):
        self.raw = raw_store
        self.semantic = semantic_store
        self.image_manager = image_manager
        self.max_raw_msg = max_raw_msg
    
    def get_search_semantic_memories(self):
        @tool
        def search_semantic_memories(
            query_text: Optional[str] = None, 
            query_image: Optional[str] = None, 
            top_k: int = 10
        ):
            """
            Search high-level semantic memories using text or image or both.
            Returns memory items, each consists of id, content and evidence raw message ID ranges.
            
            Args:
                query_text: Query text
                query_image: Query image. Use image token to refer to image(e.g. <image23>). IMPORTANT: Image tokens are ONLY allowed in this field and MUST NOT appear in others. Set query_image=None if this memory contains no image.
                top_k: Number of results to return
            """
            if query_image and query_image != 'N/A':
                query_image = self.image_manager.image_token_to_image(query_image)
            
            results = self.semantic.hybrid_search(
                query_text=query_text,
                query_image_path=query_image,  # For prototype, text-only
                top_k=top_k
            )
            
            if not results:
                return "No relevant semantic memories found."
            
            output = []
            images = []
            for i, mem in enumerate(results):
                json_msg = {
                    "id": mem.memory_id,
                    "text": mem.text or 'N/A',
                    "image": "<image>" if mem.image_path else 'N/A',
                    "Evidence IDs": mem.evidence_ids
                }
                output.append(json_msg)
                if mem.image_path:
                    images.append(mem.image_path)
            
            return self.image_manager.format_obj_to_content(output, images)
        
        return search_semantic_memories
    
    def get_fetch_raw_messages(self):
        @tool
        def fetch_raw_messages(id_ranges: str) -> list[dict]:
            """
            Fetch raw messages by ID ranges. 
            
            WARNING: Use with caution - this tool can return LARGE amounts of data
            that may exceed context limits and impact performance.
            
            Best practices:
            1. Only call this tool when you DO need to examine raw messages
            2. Always estimate result size before calling
            3. Use narrow, specific ranges when possible
            4. Consider iterative fetching for large ranges when necessary
            5. Only request data actually needed for the current task
            6. Give ranges in order.
            
            Args:
                id_ranges: JSON string of ranges, e.g. "[[1,5], [12,12]]"
            """
            try:
                ranges = json.loads(id_ranges)
                messages = self.raw.fetch_by_ids(ranges)
                
                if not messages:
                    return f"No messages found in ranges {id_ranges}"
                
                output = [{
                    "type": "text", "text": f"{min(len(messages), self.max_raw_msg)} messages fetched" + (
                        f", truncated to {self.max_raw_msg}:\n\n" if len(messages) > self.max_raw_msg else ":\n\n"
                    )
                }]
                images = []
                for msg in messages[:self.max_raw_msg]:
                    output.append({
                        "id": msg.msg_id,
                        "timestamp": msg.timestamp.strftime(TIME_FMT),
                        "speaker": msg.role,
                        "text": msg.text or 'N/A',
                        "image": "<image>" if msg.image_path else 'N/A'
                    })
                    if msg.image_path:
                        images.append(msg.image_path)
                              
                return self.image_manager.format_obj_to_content(output, images)

            except Exception as e:
                # print(e)
                # print(f"Error parsing ID ranges: {id_ranges}")
                return f"Error parsing ID ranges: {id_ranges}"
    
        return fetch_raw_messages
    
    def get_fetch_raw_messages_by_time(self):
        @tool
        def fetch_raw_messages_by_time(start_date: str, end_date: str) -> str:
            """
            Fetch raw messages within a time range.
            
            Args:
                start_date: ISO format {(YYYY-MM-DD)}
                end_date: ISO format (YYYY-MM-DD)
            """
            try:
                start = datetime.fromisoformat(start_date)
                end = datetime.fromisoformat(end_date)
                messages = self.raw.fetch_by_timerange(start, end)
                
                if not messages:
                    return f"No messages between {start_date} and {end_date}"
                
                output = [f"Found {len(messages)} messages:\n"]
                for msg in messages[:20]:  # Limit output
                    output.append(f"[{msg.msg_id}] {msg.timestamp.strftime(TIME_FMT)} - {msg.role}: {msg.text[:50]}...")
                
                return "\n".join(output)
            except Exception as e:
                # print(e)
                # print("Error parsing dates")
                return f"Error parsing dates"
        
        return fetch_raw_messages_by_time

    def get_add_memory(self):
        @tool
        def add_memory(
            text: str,
            image: Optional[str] = None,
            image_caption: Optional[str] = None,
            evidence_ids: str = "[]",
        ) -> str:
            """
            Create a new semantic memory entry.
            For optional args, set to None if you don't need them.
            
            Args:
                text: Memory text content
                image: Memory image content. Use image token to refer to image(e.g. <image23>). IMPORTANT: Image tokens are ONLY allowed in this field and MUST NOT appear in others. Set image=None if this memory contains no image.
                image_caption: Caption for associated images
                evidence_ids: JSON string of ID ranges, e.g. "[[1,3], [5,5]]"
            """
            try:
                ev_ids = json.loads(evidence_ids)
                if image == 'N/A':
                    image = None
                if image_caption == 'N/A':
                    image_caption = None
                memory = SemanticMemory(
                    # memory_id=str(uuid.uuid4()),
                    text=text,
                    image_caption=image_caption if image_caption else None,
                    image_path=self.image_manager.image_token_to_image(image),
                    evidence_ids=ev_ids,
                )
                mem_id = self.semantic.add(memory)
                return f"Created memory, id: {mem_id}"
            except Exception as e:
                # print(e)
                # print(f"Error creating memory: {e}")
                return f"Error creating memory: {e}"

        return add_memory
        
    def get_delete_memory(self):
        @tool
        def delete_memory(memory_id: str) -> str:
            """
            Delete a semantic memory.
            
            Args:
                memory_id: ID of memory to delete
            """
            success = self.semantic.delete(memory_id)
            if success:
                return f"Deleted memory {memory_id}"
            else:
                return f"Memory {memory_id} not found"

        return delete_memory
   
@dataclass
class MemoryManagerState:
    """State for MemoryManager graph"""
    messages: list[BaseMessage]
    
    # Input from ChatAgent
    operation: Literal["query", "update"] = "query"
    query_text: Optional[str] = None
    query_image_path: Optional[str] = None
    
    context: list[RawMessage] = field(default_factory=list)
    
    # Iterative retrieval state
    messages: list[BaseMessage] = field(default_factory=list)
    iteration_count: int = 0
    
    response: str | list[dict] = ""


class MemoryManager:
    max_iteration: int = 15
    tools: dict[str, BaseTool]
    
    def __init__(
        self,
        tools: MemoryManagerTools,
        raw_store: RawMessageStore,
        semantic_store: SemanticStore,
        llm: ChatOpenAI,
        image_manager: ImageManager,
        config: MemoryManagerConfig
    ):
        # self.tools = tools
        self.raw_store = raw_store
        self.semantic_store = semantic_store
        self.tools = {
            "search_semantic_memories": tools.get_search_semantic_memories(),
            "fetch_raw_messages": tools.get_fetch_raw_messages(),
            "fetch_raw_messages_by_time": tools.get_fetch_raw_messages_by_time(),
            
            "add_memory": tools.get_add_memory(),
            "delete_memory": tools.get_delete_memory(),
        }
        self.tool_cls = tools
        self.config = config
        
        self.image_manager = image_manager
        self.llm = llm
        self.graph = self._build_graph()
        
    def _prepair_context(self, context: list[RawMessage]) -> list[dict]:
        content = [{
            "type": "text", "text": "[\n"
        }]
        for msg in context:
            chunks = self.image_manager.format_to_msg_content(
                text=msg.text,
                image=msg.image_path,
                speaker=msg.role,
                timestamp=msg.timestamp.strftime(TIME_FMT),
                message_id=msg.msg_id
            )
            content += chunks + [{
                "type": "text", "text": ",\n"
            }]
        
        content.append({
            "type": "text", "text": "]\n"
        })
        return content
            
    def _fill_query_sys_prompt(self, state: MemoryManagerState) -> MemoryManagerState:
        sys_prompt = """You are MemoryManager handling a QUERY request from ChatAgent.

=== GOAL ===
Retrieve relevant information and return a focused answer to the query.
Use image tokens (e.g. <image23>) when including images in your response.


=== EXECUTION FLOW ===
1. SEARCH semantic memories (always start here)
2. ASSESS if results answer the query:
   - Sufficient? → Proceed to step 4
   - Need details? → Fetch raw messages using evidence_ids
3. ITERATE if needed (additional searches/fetches)
4. RESPOND with a concise, query-focused answer and provide neccessary context.


IMPORTANT:
1. If there are time references (e.g. "last year", "2 months ago"), 
    fitst reason and determine what's the reference time, then calculate the actual date based on the timestamp.
    e.g. (message on July 5th, send by Mary) I went for a trip yesterday -> reference time: July 5th -> actual trip day: July 4th.
    Assume current time is the last messege's time.
2. Always convert relative time references to specific dates, months or years.
3. For optional arguments when calling tools, DO NOT give them in tool call args dict when you don't need them. (i.e. don't fill with 'N/A')


=== EXAMPLES ===
Query: "When did Jack visit Tokyo?"
→ search_semantic_memories("Jack Tokyo visit")
→ Respond: "Jack visited Tokyo in March 2024"

Query: "What did Jack say about the restaurant yesterday?"(current time: Jun 9th)
→ search_semantic_memories("Jack restaurant Jun 8th")
→ fetch_raw_messages using evidence_ids (need exact words)
→ Respond: [Jack's specific comments]

Query: "What's Sarah's cat's name and can you show me?"
→ search_semantic_memories("Sarah cat")
→ Respond: "Sarah's cat is named Whiskers, shown here: <image12>"


Most recent chat messages: 
---
<context>
---
Current query: <query>
"""
        
        # 1. prepair messages
        sys_prompt_content = []
        chunks = re.split(r'(<context>|<query>)', sys_prompt)
        for chunk in chunks:
            if chunk == "<context>":
                sys_prompt_content += self._prepair_context(state.context)
            elif chunk == "<query>":
                sys_prompt_content += self.image_manager.format_to_msg_content(
                    text=state.query_text,
                    image=state.query_image_path
                )
            else:
                sys_prompt_content.append({
                "type": "text", "text": chunk
            })
        
        state.messages = [HumanMessage(sys_prompt_content)]
        return state
        
    def _handle_query(
        self, 
        state: MemoryManagerState
    ) -> Command:
        messages = state.messages
        
        response = self.llm.bind_tools([
            self.tools["search_semantic_memories"],
            self.tools["fetch_raw_messages"],
            self.tools["fetch_raw_messages_by_time"],      
        ], parallel_tool_calls=False).invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            try:
                response = self.image_manager.format_msg_to_content(
                    response.content
                )
            except Exception as e:
                messages.append(HumanMessage(content=str(e)))
                return Command(
                    update={"messages": messages},
                    goto="handle_query"
                )
            return Command(
                update={"messages": messages, "response": response},
                goto=END
            )
        
        return Command(
            update={"messages": messages},
            goto="exec_tool"
        )
    
    def _exec_tool(
        self, 
        state: MemoryManagerState
    ) -> Command[Literal['handle_query', 'handle_update']]:
        last_message: AIMessage = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state

        for tool_call in last_message.tool_calls:
            state.iteration_count += 1
            if state.iteration_count == self.max_iteration:
                print("MAX_ITERATION_REACHED")
                state.messages.append(ToolMessage(
                    content="Tool call failed: Tool call count limit exceeded!",
                    tool_call_id=tool_call["id"]
                ))
                return state
            
            try:
                query_image = tool_call["args"].get("query_image")
                if query_image == 'N/A':
                    query_image = None
                query_image = self.image_manager.image_token_to_image(query_image)
            except Exception as e:
                state.messages.append(ToolMessage(
                    content=f"Tool error: {e}",
                    tool_call_id=tool_call["id"]
                ))
                return state
                
            # self.tools_cls.tool_call_id = tool_call["id"]
            try:
                tool_result = self.tools[tool_call["name"]].invoke(input=tool_call["args"])
            except Exception as e:
                tool_result = f"Tool error: Please check your input and try again. ({str(e)})"
            
            # memory_result = self._prepair_tool_result(memory_result)
            
            state.messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call['id'])
            )
        
        update = {
            "messages": state.messages,
            "iteration_count": state.iteration_count
        }
        if state.operation == 'query':
            return Command(
                update=update,
                goto="handle_query"
            )
        return Command(update=update,goto="handle_update")
    
    def _fill_update_sys_prompt(self, state: MemoryManagerState) -> MemoryManagerState:
        sys_prompt = """You are MemoryManager processing an UPDATE request from ChatAgent.

=== YOUR ROLE ===
Analyze ChatAgent's update suggestion and maintain the semantic memory database.
You have access to:
- Semantic memory DB: High-level, structured memories you maintain
- Raw message DB: Original conversation messages (read-only)

=== EXECUTION FLOW ===

Step 1: UNDERSTAND THE REQUEST
- Parse ChatAgent's suggestion
- Identify key entities, events, timeframes, and relationships
- Determine if this is: new information / update / correction / summarization

Step 2: QUERY EXISTING MEMORY
- Search semantic memories for related content
- If needed, fetch raw messages for context (use sparingly - see tool warnings)
- Identify: duplicates, contradictions, gaps, related memories. If there are contradictions,
    prioritize the most recent one.

Step 3: PLAN OPERATIONS
Decide on operation type(s):
- CREATE: Novel information not present in memory
- DELETE: Outdated, contradicted, or now-subsumed information
- BOTH: When updating (delete old + add refined version)
- NONE: If information already adequately captured

Step 4: EXECUTE & VERIFY
- Perform planned operations
- For complex updates, you may need multiple add_memory calls to capture different granularities
- Verify no critical information was lost

Step 5: COMPLETE
- When done, respond ONLY with: ""

IMPORTANT:
[Temporal Processing]
- ALWAYS resolve relative time to absolute dates
  ✓ Message: "went hiking yesterday" (Jan 7, 2022) → Store: "Jan 6, 2022"
  ✓ "last summer" (Nov 2023) → "Summer 2023" or "Jun-Aug 2023"
  ✓ "two months ago" (Dec 15, 2024) → "mid-October 2024" or "Oct 15, 2024"
- Include time context in memories unless the information is truly timeless
- Preserve original timestamp precision when available

[Entity References]
- Use SPECIFIC NAMES, never pronouns (he/she/they) or generic terms (user/person)
  ✓ "Jack prefers morning coffee"
  ✗ "He prefers morning coffee"
  ✗ "User prefers morning coffee"
- When speaker name is ambiguous, use context or evidence_ids to clarify

[Granularity Strategy]
- Break complex information into atomic + summary memories:
  Example: "Jack's Tokyo trip (Mar 2025): cherry blossoms, temples, budget travel"
  → Add multiple memories:
    1. "Jack planning Tokyo trip in March 2025" (summary)
    2. "Jack interested in cherry blossom viewing during Tokyo trip"
    3. "Jack wants to visit traditional temples in Tokyo"
    4. "Jack is budget-conscious for Tokyo trip"

[Evidence Tracking]
- ALWAYS provide evidence_ids linking memories to source messages
- Combine contiguous ranges

Most recent chat messages: <context>

ChatAgent suggests: <query>
"""
        sys_prompt_content = []
        chunks = re.split(r'(<context>|<query>)', sys_prompt)
        for chunk in chunks:
            if chunk == "<context>":
                sys_prompt_content += self._prepair_context(state.context)
            elif chunk == "<query>":
                sys_prompt_content += self.image_manager.format_to_msg_content(
                    text=state.query_text,
                    image=state.query_image_path
                )
            else:
                sys_prompt_content.append({
                "type": "text", "text": chunk
            })
        
        state.messages = [HumanMessage(sys_prompt_content)]
        return state
    
    def _handle_update(self, state: MemoryManagerState) -> Command:
        messages = state.messages

        response = self.llm.bind_tools([
            self.tools["search_semantic_memories"],
            self.tools["fetch_raw_messages"],
            self.tools["fetch_raw_messages_by_time"],   
            
            self.tools["add_memory"],
            self.tools["delete_memory"],
        ], parallel_tool_calls=False).invoke(messages)
        
        messages.append(response)
        if not response.tool_calls:
            return Command(
                update={"messages": messages, "response": response.content},
                goto=END
            )
        
        return Command(
            update={"messages": messages},
            goto="exec_tool"
        )
    
    def _build_graph(
        self
    ) -> CompiledStateGraph[MemoryManagerState, None, MemoryManagerState, MemoryManagerState]:
        """Build LangGraph workflow for MemoryManager"""
        workflow = StateGraph(MemoryManagerState)
        
        workflow.add_node(END, lambda x: x)
        workflow.add_node("route", self._route_operation)
        workflow.add_node("fill_query_sys_prompt", self._fill_query_sys_prompt)
        workflow.add_node("handle_query", self._handle_query)
        workflow.add_node("exec_tool", self._exec_tool)
        workflow.add_node("fill_update_sys_prompt", self._fill_update_sys_prompt)
        workflow.add_node("handle_update", self._handle_update)
        
        
        workflow.set_entry_point("route")
        
        workflow.add_conditional_edges(
            "route",
            lambda s: s.operation,
            {
                "query": "fill_query_sys_prompt",
                "update": "fill_update_sys_prompt"
            }
        )
        
        workflow.add_edge("fill_query_sys_prompt", "handle_query")
        workflow.add_edge("fill_update_sys_prompt", "handle_update")
        
        return workflow.compile()
    
    def query(
        self, 
        context: list[RawMessage],
        query_text: Optional[str] = None, 
        query_image: Optional[str] = None,
    ) -> list[dict]:
        """Handle memory query from ChatAgent"""
        state = MemoryManagerState(
            operation="query",
            query_text=query_text,
            query_image_path=query_image,
            context=context
        )
        result = self.graph.invoke(state,config={
                # "callbacks": [handler],
                "timeout": 20,
                # "callbacks": [langfuse_handler] if DEBUG else [],
                "configurable": {"thread_id": "1"}
            },)
        return result["response"]
    
    def update(
        self, 
        context: list[RawMessage],
        query_text: Optional[str] = None, 
        query_image: Optional[str] = None,
    ) -> str:
        """Handle memory update from ChatAgent"""
        state = MemoryManagerState(
            operation="update",
            query_text=query_text,
            query_image_path=query_image,
            context=context
        )
        result = self.graph.invoke(
            state,
            config={
                # "callbacks": [handler],
                # "callbacks": [langfuse_handler] if DEBUG else [],
                "configurable": {"thread_id": "1"}
            },)
        return result["response"]
    
    def _route_operation(self, state: MemoryManagerState) -> MemoryManagerState:
        """Route to query or update branch"""
        return state