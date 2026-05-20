from dataclasses import dataclass
from datetime import datetime
from tkinter import END
from typing import Literal, Optional
from langchain.tools import tool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from ..stores import RawMessage, RawMessageStore, ImageManager
from .memory_manager import MemoryManager
from ..config import ChatAgentConfig
from ..utils.message import encode_image_to_base64

@dataclass
class ChatAgentState:
    """State for ChatAgent graph"""
    messages: list[BaseMessage]
    
    # Last User input
    user_message: str = ""
    user_image_path: Optional[str] = None
    
    
    # Memory interaction
    query_iteration: int = 0
    update_iteration: int = 0
    
    # Output
    response: str = ""



class ChatAgent:
    llm: ChatOpenAI
    config: ChatAgentConfig
    memory_manager_context_window: int = 5
    raw_messages: list[RawMessage]
    # for llm
    chat_messages: list[BaseMessage]
    system_prompt: str = None

    def __init__(
        self,
        memory_manager: MemoryManager,
        raw_store: RawMessageStore,
        llm: ChatOpenAI,
        image_manager: ImageManager,
        config: ChatAgentConfig,
        update_memory: bool = True,
        update_only: bool = False,
    ):
        # validate params
        if update_only and not update_memory:
            import warnings
            warnings.warn("`update_only` set to True but `update_memory` is not enabled! Overide to True.")
            update_memory = True

        self.memory_manager = memory_manager
        self.raw_store = raw_store
        self.llm = llm
        self.image_manager = image_manager
        self.raw_messages = []
        self.update_memory = update_memory
        self.config = config

        self.tools = {
            "query": self._create_query_tool(),
            "update": self._create_update_tool(),
        }
        self.graph = self._build_graph(update_only)
        self.chat_messages = []

    def init_conversation(self, system_prompt: Optional[str] = None):
        """Initialize conversation for evaluation mode"""
        self.chat_messages = []
        self.raw_messages = []
        self.system_prompt = system_prompt
        if system_prompt:
            self.chat_messages.append(SystemMessage(system_prompt))
        else:
            # Default system message
            self.chat_messages.append(SystemMessage("""
You are an AI assistant with access to long-term memory.
"""))

    def _prepair_message_content(self, content: list[str]) -> list[str]:
        res = []
        for x in content:
            if x["type"] == "text":
                res.append(x)
            elif x["type"] == "image":
                image_path = x["url"]
                image_token = self.image_manager.image_to_image_token(image_path)
                res.append({"type": "text", "text": f"{image_token}: "})
                x['url'] = encode_image_to_base64(image_path)
                res.append(x)
        return res
    
    def _create_query_tool(self):
        @tool
        def query_memory(
            # state: Annotated[dict, InjectedState],
            text: Optional[str]=None,
            image: Optional[str]=None,
        ) -> list[dict]:
            """
            Send a memory query request to the Memory Manager agent.

            CRITICAL RULES:
            - Query memory only when past, persistent, or non-local context is required.
            - Do NOT query memory if the answer can be derived from the recent conversation.
            - If uncertain whether memory is needed, DO NOT query.

            WHEN to query memory:
            - The user asks about past events, history, prior actions, prior
              outcomes, prior tool-calls, or any observation recorded earlier.
            - The user references entities, objects, locations, or events not
              present in the recent context.
            - The user asks about stored facts, relationships, identities,
              preferences, task state, or environment history.
            - Temporal or historical questions
            (e.g. “When did X happen?”, “Have we discussed Y before?”).

            WHEN NOT to query memory:
            - Casual greetings or small talk.
            - Questions answerable using only the recent dialogue.
            - Requests that require reasoning or generation but no factual recall.
            - No persistent or historical information is involved.
            
            IMPORTANT: For optional arguments, DO NOT give them in args when you don't need them. (i.e. don't fill with 'N/A')
            
            ARGUMENTS:
            - text (required):
            A concise, natural-language command describing what information
            should be retrieved from memory.
            Phrase this as an instruction to the Memory Manager
            (what to look for, who or what is involved, and any relevant time scope).
            MUST NOT contain any image tokens.

            - image (optional):
            A reference to a single image relevant to the query.
            MUST use the format "<image{id}>".
            Image tokens are ONLY allowed in this field and MUST NOT appear in `text`.

            VALID EXAMPLES:
            query_memory(
                text="Retrieve any stored information about the man shown in the image.",
                image="<image3>"
            )

            query_memory(
                text="Find records of where the user traveled last summer."
            )

            OUTPUT:
            - Return relevant memory context distilled by the Memory Manager.
            - Returned information may be multimodal and can include text and images.
            """
        
        return query_memory        
            
    def _query_memory(
        self, 
        state: ChatAgentState
    ) -> ChatAgentState:
        """Execute query_memory tool calls"""
        last_message = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
        
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == "query_memory":
                state.query_iteration += 1
                if state.query_iteration == self.config.max_query_iteration:
                    print("MAX_ITERATION_REACHED")
                    state.messages.append(ToolMessage(
                        content="Tool call failed: generate response state max tool call count limit exceeded!",
                        tool_call_id=tool_call["id"]
                    ))
                    return state
                
                tool_call_id = tool_call["id"]
                query_text = tool_call['args'].get('text')
                query_image = tool_call['args'].get('image')
                if query_image == 'N/A':
                    query_image = None

                try:
                    query_image = self.image_manager.image_token_to_image(query_image)
                except Exception as e:
                    state.messages.append(ToolMessage(
                        content=f"Tool error: {e}",
                        tool_call_id=tool_call_id
                    ))
                    return state
                    
                # Call MemoryManager
                context = self.raw_messages[-self.memory_manager.config.context_window:]
                memory_result = self.memory_manager.query(
                    query_text=query_text,
                    query_image=query_image,
                    context=context
                )
                
                state.messages.append(
                    ToolMessage(content=memory_result, tool_call_id=tool_call['id'])
                )
        
        return state
    
    def _generate_response(
        self, 
        state: ChatAgentState
    ) -> Command[Literal["query_memory", "respond"]]:
        # last user message should already be added to messages
        messages = trim_messages(
            state.messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=8000,
            include_system=True,
            start_on="human",
            # end_on=("human", "tool")
        )
        # assert messages[-1].type == 'human'
        llm = self.llm
        if state.query_iteration < self.config.max_query_iteration:
            # only allow tool call if not reach max_query_iteration
            response = llm.bind_tools([self.tools["query"]]).invoke(messages)
        else:
            response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            # reset query call counts
            return Command(
                update={"messages": messages, "query_iteration": 0},
                goto="respond"
            )
        
        return Command(
            update={"messages": messages},
            goto="query_memory"
        )
    
    def _respond(self, state: ChatAgentState) -> ChatAgentState:
        # assert state.messages[-1].name == 'ai'
        
        state.response = state.messages[-1].content
        return state
    
    def _create_update_tool(self):
        @tool
        def update_memory(
            # state: Annotated[dict, InjectedState],
            text: Optional[str]=None,
            image: Optional[str]=None,
        ) -> list[dict]:
            """
            Send a memory update instruction to the Memory Manager agent.

            CRITICAL RULES:
            - DO NOT duplicate existing memories.
            - DEFAULT TO UPDATE: When uncertain, prefer updating over skipping.
            - Every incoming user message in an agent / task / tool-use
              trajectory (observations, actions, feedback, plans, intermediate
              results) is information-bearing and SHOULD be recorded — do not
              dismiss these as "transient state". Err on the side of calling
              update_memory at least once per turn unless the message is pure
              phatic chat.

            WHEN to update memory:
            # NOTE: this prompt has been broadened from the upstream's
            # "Personal facts" scoping — on VAB-MM / agent-task data the
            # narrow personal-assistant framing rejected every agent
            # observation, giving memory_recall = 0%.  The generalised
            # categories below restore coverage without changing code.
            - Any durable factual observation from the conversation that may be
              referenced later: entities, objects, locations, events, states,
              actions, outcomes, tool-calls, agent observations, task steps,
              environment feedback, errors, results.
            - Personal facts (when applicable): Names, relationships, roles,
              identities, preferences, habits, travel plans, scheduled events.
            - Temporal markers: Dates, times, event sequences, duration mentions.
            - Stable factual information that should persist across turns/sessions.
            - Summarize over a conversation session
                - Session boundary: Detect topic shifts OR >30min gaps in timestamps
                - Coverage goal: Every message(except cases mentioned in "NOT TO UPDATE" below) 
                    should be captured in at least one update
                - Summarization depth:
                    - Single session: Capture key points, decisions, and action items
                    - Weekly span: Major themes and significant events only
                    - Monthly/yearly: High-level milestones and changes
                - DO NOT: Simply restate each message verbatim
            - If you are not sure whether to update current message or not, you can first skip and then look back and decide
            whether not update or make a single update focused on that or make an update of summarization on several past messages
            as a whole based on the context.

            WHEN NOT to update memory:
            - Casual conversation, greetings, or social niceties.
            - Information already present in memory (avoid restating).

            ARGUMENTS:
            - text (optional):
            Natural-language instruction for the memory operation.
                - Be specific and factual
                - Include temporal context when available
                - MUST NOT contain image tokens (use <imageN> format)

            - image (optional):
            Reference to an image using EXACTLY "<image{id}>" format.
                - ONLY allowed in this field
                - Required when visual information is essential to the memory
                - If not needed, set to None or simply NOT include it in calling args dict. DO NOT set to anything else(e.g. 'N/A' or '')

            ## GOOD:
            update_memory(
                text="Record: Jack (left person in image) is user's college roommate, currently works at Google (mentioned 2024-06-10).",
                image="<image3>"
            )

            update_memory(
                text="Session summary (Jan 15-18): User planning Tokyo trip for March 2025. Interested in cherry blossoms, traditional temples. Budget conscious. Friend Sarah may join."
            )

            update_memory(
                text="Temporal update: User's meeting with client rescheduled from Jan 20 to Jan 27 due to client's illness."
            )

            ## BAD:
            update_memory(text="User talked about something")  # Too vague
            update_memory(text="Hi")  # Phatic, no information
            update_memory(text="User likes Tokyo and <image5>")  # Image token in wrong field


            OUTPUT:
            - Return a brief description of the update result from the Memory Manager.
            """
        
        return update_memory 
    
    def _update_stage(
        self, 
        state: ChatAgentState
    ) -> Command:
        # last user message should already be added to messages
        messages = trim_messages(
            state.messages,
            strategy="last",
            token_counter=count_tokens_approximately,
            max_tokens=8000,
            include_system=True,
            start_on="human",
            # end_on=("human", "tool")
        )
        
        llm = self.llm
        if state.update_iteration < self.config.max_update_iteration:
            # only allow tool call if not reach max_query_iteration
            response = llm.bind_tools(
                [
                    self.tools["query"],
                    self.tools["update"]
                ], 
                parallel_tool_calls=False
            ).invoke(messages)
        else:
            response = llm.invoke(messages)
        
        messages.append(response)
        if not response.tool_calls:
            return Command(
                update={"messages": messages},
                goto=END
            )
        
        return Command(
            update={"messages": messages},
            goto="exec_update_stage_tools"
        )
    
    def _exec_update_state_tools(
        self, 
        state: ChatAgentState
    ) -> ChatAgentState:
        """Execute update state tool calls"""
        last_message = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            return state
 
        for tool_call in last_message.tool_calls:
            if tool_call['name'] == "query_memory":
                state.update_iteration += 1
                if state.update_iteration == self.config.max_update_iteration:
                    print("MAX_ITERATION_REACHED")
                    state.messages.append(ToolMessage(
                        content="Tool call failed: update state max tool call count limit exceeded!",
                        tool_call_id=tool_call["id"]
                    ))
                    return state
                
                tool_call_id = tool_call["id"]
                query_text = tool_call['args'].get('text')
                query_image = tool_call['args'].get('image')
                if query_image == 'N/A':
                    query_image = None

                try:
                    query_image = self.image_manager.image_token_to_image(query_image)
                except Exception as e:
                    state.messages.append(ToolMessage(
                        content=f"Tool error: {e}",
                        tool_call_id=tool_call_id
                    ))
                    return state
                    
                # Call MemoryManager
                context = self.raw_messages[-self.memory_manager.config.context_window:]
                memory_result = self.memory_manager.query(
                    query_text=query_text,
                    query_image=query_image,
                    context=context
                )
                memory_result = self._prepair_message_content(memory_result)
                
                state.messages.append(
                    ToolMessage(content=memory_result, tool_call_id=tool_call['id'])
                )

            elif tool_call['name'] == "update_memory":
                state.update_iteration += 1
                if state.update_iteration == self.config.max_update_iteration:
                    print("MAX_ITERATION_REACHED")
                    state.messages.append(ToolMessage(
                        content="Tool call failed: update state max tool call count limit exceeded!",
                        tool_call_id=tool_call["id"]
                    ))
                    return state
                
                tool_call_id = tool_call["id"]
                query_text = tool_call['args'].get('text')
                query_image = tool_call['args'].get('image')
                if query_image == 'N/A':
                    query_image = None
                
                try:
                    query_image = self.image_manager.image_token_to_image(query_image)
                except Exception as e:
                    state.messages.append(ToolMessage(
                        content=f"Tool error: {e}",
                        tool_call_id=tool_call_id
                    ))
                    return state
                    
                
                # Call MemoryManager
                context = self.raw_messages[-self.memory_manager.config.context_window:]
                update_result = self.memory_manager.update(
                    context=context,
                    query_text=query_text,
                    query_image=query_image,
                )
                
                state.messages.append(
                    ToolMessage(content=update_result, tool_call_id=tool_call['id'])
                )
        
        return state
    
    def _build_graph(
        self, update_only: bool
    ) -> CompiledStateGraph[ChatAgentState, None, ChatAgentState, ChatAgentState]:
        """Build LangGraph workflow for ChatAgent"""
        workflow = StateGraph(ChatAgentState)
        
        workflow.add_node(END, lambda x: x)
        if not update_only:
            workflow.set_entry_point("generate_response")
            workflow.add_node("generate_response", self._generate_response)
            workflow.add_node("query_memory", self._query_memory)
            workflow.add_node("respond", self._respond)
                
            workflow.add_edge("query_memory", "generate_response")
            
            if not self.update_memory:
                workflow.add_edge("respond", END)
            else:
                workflow.add_edge("respond", "update_stage")
        
        if self.update_memory:
            workflow.add_node("update_stage", self._update_stage)
            if update_only:
                workflow.set_entry_point("update_stage")
            workflow.add_node("exec_update_stage_tools", self._exec_update_state_tools)
            workflow.add_edge("exec_update_stage_tools", "update_stage")
            
        return workflow.compile()
    
    def _prepair_user_message(
        self,
        user_text: Optional[str], 
        user_image_path_or_url: Optional[str]
    ) -> list[dict]:
        """
        Prepair the user message in openai api style, including handling image.
        
        Returns:
            content
        """
        content = []
        if user_text:
            content.append({"type":"text", "text":user_text})
        if user_image_path_or_url:
            content.append({"type": "image", "url": user_image_path_or_url})
            
        return self._prepair_message_content(content)
    
    def chat(
        self, 
        user_text: Optional[str] = None, 
        user_image_path_or_url: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        role: Optional[str] = None
    ) -> str:
        """Main chat interface"""
        # 1. write user message to RawStore first
        user_msg = RawMessage(
            msg_id=None,
            timestamp=timestamp or datetime.now(),
            role=role or "user",
            text=user_text,
            image_path=user_image_path_or_url
        )
        msg_id = self.raw_store.append(user_msg)
        user_msg.msg_id = msg_id
        
        # 2. add new raw user message to self.raw_messages
        self.raw_messages.append(user_msg)
        
        # 3. init state for this turn
        messages = self.chat_messages + [
            HumanMessage(
                content=self._prepair_user_message(user_text, user_image_path_or_url)
            )
        ]
        state = ChatAgentState(
            messages=messages,
            user_message=user_text,
            user_image_path=user_image_path_or_url
        )
        
        # 4. run graph
        result = self.graph.invoke(
            state, 
            # config={
            #     # "callbacks": [handler],
            #     # "callbacks": [langfuse_handler] if DEBUG else [],
            #     # "configurable": {"thread_id": "1"}
            # },
        )
        
        # self.chat_messages += [
        #     AIMessage(result['response'])
        # ]
        self.chat_messages = result['messages']
        
        # 5. Store assistant response
        self.raw_store.append(RawMessage(
            msg_id=None,
            timestamp=timestamp or datetime.now(),
            role="assistant",
            text=result["response"]
        ))
        
        return result["response"]
