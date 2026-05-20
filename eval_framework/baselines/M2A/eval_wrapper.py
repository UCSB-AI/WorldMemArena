"""
Evaluation Wrapper for M2A System

This module provides a wrapper class that implements the evaluation interface
used by the data/evaluator.py module.

The wrapper adapts the M2ASystem to work with the evaluation pipeline,
handling conversation initialization, chat processing, and question answering.
"""
from typing import Optional
from pathlib import Path
from datetime import datetime
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from agent.m2a import M2ASystem, M2AConfig
from agent.config import TIME_FMT
from agent.agents.chat_agent import ChatAgent

class M2AEvaluationWrapper:
    """
    Evaluation wrapper for M2A system.

    This class implements the interface expected by the evaluator:
    - start_conversation(conv_info) - Initialize for a new conversation
    - chat(dialogue) - Process dialogue history
    - question(text, image) - Answer a question about the conversation
    - over(**kwargs) - Save evaluation results
    """

    def __init__(
        self,
        config: Optional[M2AConfig] = None,
        db_dir: str = "eval_results",
    ):
        """
        Initialize the M2A evaluation wrapper.

        Args:
            config: M2AConfig object. If None, loads from environment variables.
            db_dir: Directory to save evaluation results and intermediate data.
        """
        self.config = config
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # Initialize M2A system
        self.m2a = M2ASystem(config=config)

        # For evaluation mode, we use a separate LLM for time formatting
        # This is a lightweight operation that doesn't need the full agent
        self.time_llm = ChatOpenAI(
            model=self.m2a.config.llm.model,
            base_url=self.m2a.config.llm.base_url,
            api_key=self.m2a.config.llm.api_key,
            temperature=0.0,
            max_tokens=50,
        )

        # Track conversation state
        self.cur_time = None

    def start_conversation(self, conv_info: dict):
        """
        Initialize for a new conversation.

        Args:
            conv_info: Dictionary containing conversation information with keys:
                - conv_idx: Conversation index
                - speaker_0: Name of speaker 0
                - speaker_1: Name of speaker 1
        """
        self.conv_info = conv_info
        self.chat_idx = conv_info['conv_idx']

        # Initialize storage for this conversation
        raw_db_path = str(self.db_dir.joinpath(str(self.chat_idx), "raw.db"))
        semantic_db_path = str(self.db_dir.joinpath(str(self.chat_idx), "semantic.db"))

        # Re-initialize stores with new paths
        from agent.stores import RawMessageStore, SemanticStore, ImageManager

        self.m2a.raw_store = RawMessageStore(
            db_path=raw_db_path,
            reuse=False,  # Always create new for each conversation for now
        )

        self.m2a.semantic_store = SemanticStore(
            db_path=semantic_db_path,
            re_use=False,
            text_embedder_config=self.m2a.config.text_embedding,
            multimodal_embedder_config=self.m2a.config.multimodal_embedding,
        )

        # Re-initialize image manager
        self.m2a.image_manager = ImageManager()

        # Re-initialize memory manager with new stores
        from agent.agents.memory_manager import MemoryManagerTools

        mm_tools = MemoryManagerTools(
            self.m2a.raw_store,
            self.m2a.semantic_store,
            self.m2a.image_manager,
            max_raw_msg=self.m2a.config.memory.max_raw_messages_return,
        )

        from agent.agents.memory_manager import MemoryManager

        self.m2a.memory_manager = MemoryManager(
            tools=mm_tools,
            raw_store=self.m2a.raw_store,
            semantic_store=self.m2a.semantic_store,
            llm=self.m2a.llm,
            image_manager=self.m2a.image_manager,
            config=self.m2a.config.memory_manager
        )

        # Re-initialize chat agent with new components

        self.m2a.chat_agent = ChatAgent(
            memory_manager=self.m2a.memory_manager,
            raw_store=self.m2a.raw_store,
            llm=self.m2a.llm,
            image_manager=self.m2a.image_manager,
            update_memory=True,
            config=self.m2a.config.chat_agent,
            update_only=True,
        )

        self.m2a.chat_agent.init_conversation(
            system_prompt=self._get_eval_system_prompt(conv_info)
        )

    def chat(self, dialogue: list[dict]):
        """
        Process a dialogue history (for building memory).

        Args:
            dialogue: List of dialogue turns, each with keys:
                - speaker: Speaker index (0 or 1)
                - dia_id: Dialogue ID
                - images: List of image paths
                - text: The message text
                - timestamp: Timestamp string
        """
        for idx, turn in tqdm(
            enumerate(dialogue), 
            total=len(dialogue),
            desc=f"[conv: {self.conv_info['conv_idx']}] CHAT"
        ):
            speaker = turn['speaker']
            dia_id = turn['dia_id']
            images = turn['images']
            text = turn['text']
            timestamp = turn['timestamp']

            # Convert timestamp
            formatted_time = self._format_time(timestamp)
            speaker_name = self.conv_info[f"speaker_{speaker}"]

            input_text = f"({speaker_name}, {formatted_time}) {text}"

            # Handle image
            image_path = None
            if images and images[0]:
                image_path = images[0]

            # Chat the input
            self.m2a.chat_agent.chat(
                user_text=input_text,
                user_image_path_or_url=image_path,
                timestamp=formatted_time,
                role=speaker_name,
            )

            # Update current time after each turn
            self.cur_time = formatted_time

    def question(
        self,
        text: Optional[str] = None,
        image: Optional[str] = None,
    ) -> str:
        """
        Answer a question about the conversation.

        Args:
            text: The question text
            image: Optional image file path

        Returns:
            The assistant's response
        """
        self.m2a.chat_agent = ChatAgent(
            memory_manager=self.m2a.memory_manager,
            raw_store=self.m2a.raw_store,
            llm=self.m2a.llm,
            image_manager=self.m2a.image_manager,
            update_memory=False,  # Don't update during question phase
            config=self.m2a.config.chat_agent,
        )
        self.m2a.chat_agent.init_conversation(f"""
You are an intelligent memory assistant tasked with retrieving accurate information
from conversation memories.

# CONTEXT
You have access to a long-term memory bank which contains memories
from two speakers('{self.conv_info['speaker_0']}' and '{self.conv_info['speaker_1']}') in a conversation.

You will then be given a question regarding the conversation and these two users.
Call tools to query memory and answer the question.

IMPORTANT:
1. If there is a question about time references (e.g. "last year", "2 months ago"),
    calculate the actual date based on the timestamp.
2. Always convert relative time references to specific dates, months or years.
3. For all questions, you MUST call query tools to examine the memory.
4. If the question is about common sense(i.e. open-domain questions), you could answer with common sense.

Be concise and accurate. Do not exaplain too much unless neccessary.
Give your answer in one sentence.
Current time: {self.cur_time}.
"""
        )

        if image and image[0]:
            image = image[0]
        else:
            image = None

        return self.m2a.chat_agent.chat(
            user_text=text,
            user_image_path_or_url=image,
        )

    def over(self, **kwargs):
        """
        Notify m2a that a full test point(conversation + QAs) is done

        Args:
            **kwargs: Dictionary containing:
                - results: Evaluation results
                - summary: Evaluation summary
        """
        # Save results to JSON files
        conv_dir = str(self.db_dir.joinpath(str(self.chat_idx)))
        conv_dir_path = Path(conv_dir)
        conv_dir_path.mkdir(parents=True, exist_ok=True)

        import json

        if 'results' in kwargs:
            with open(conv_dir_path.joinpath("results.json"), 'w') as f:
                json.dump(kwargs['results'], f, indent=2)

        if 'summary' in kwargs:
            with open(conv_dir_path.joinpath("summary.json"), 'w') as f:
                json.dump(kwargs['summary'], f, indent=2)

        # Dump databases
        self.m2a.raw_store.dump_db(str(conv_dir_path.joinpath("raw.json")))
        self.m2a.semantic_store.dump_db(str(conv_dir_path.joinpath("semantic.json")))

        # Save image manager state
        image_manager_path = str(conv_dir_path.joinpath("image_manager.json"))
        self.m2a.image_manager.save(image_manager_path)

        print(f"Results saved to {conv_dir}")

    def _format_time(self, timestamp: str) -> datetime:
        """
        Format timestamp string to datetime object.

        This uses a lightweight LLM to convert the timestamp format
        to the standardized TIME_FMT format.
        """
        from langchain_core.messages import HumanMessage

        msgs = [
            HumanMessage(f"""
You are a time format converter. I will give you a time string. Your task is to convert it to format: '{TIME_FMT}'.

Rules:
1. Output ONLY the formatted time string, nothing else.
2. If the input time is not precise enough (missing hour/minute), assume the earliest possible time:
   - Missing hour/minute: use '00:00'
   - Only date: use that date at '00:00'
   - Only month: use 1st of that month at '00:00'
   - Only year: use January 1st of that year at '00:00'
3. Use Gregorian calendar.
4. If the time is ambiguous (like "2024-9-2"), interpret it as year-month-day.
5. Always output in the format: YYYY-MM-DD HH:MM (with leading zeros for single-digit months/days/hours/minutes).
6. Output the date string only and nothing else, DO NOT wrap it with a box.

Examples:
- "2024-9-2" → "2024-09-02 00:00"
- "September 2, 2024" → "2024-09-02 00:00"
- "2024-09-02 14:30" → "2024-09-02 14:30"
- "2024/09/02" → "2024-09-02 00:00"
- "2024-09" → "2024-09-01 00:00"
- "2024" → "2024-01-01 00:00"

Now process this input: {timestamp}
""")
        ]

        res = self.time_llm.invoke(msgs)
        # Extract just the formatted time
        formatted_time = res.content.strip()
        return datetime.strptime(formatted_time, TIME_FMT)

    def _get_eval_system_prompt(self, conv_info: dict) -> str:
        """
        Get the evaluation system prompt.

        Args:
            conv_info: Conversation info with speaker names

        Returns:
            System prompt for evaluation mode
        """
        speaker_0 = conv_info.get('speaker_0', 'Speaker 0')
        speaker_1 = conv_info.get('speaker_1', 'Speaker 1')

        return f"""
You are an AI memory evolution agent with access to long-term memory.

You will be given a long conversation between two users, which may also span across a
long range of time. Each message will be formated as "(speaker_name, msg_time) actual_msg_content".

IMPORTANT:
    - You DO NOT need to reply! When no tool call needed, respond with ''.
    - When updating memories, use speaker_name to specify character(DO NOT use 'user'). Use specific names,
        not pronouns(do not use he/she).
    - When updating memories, calculate the actual date based on timestamp given
        if there is time references (like "last year", "2 months ago").
        e.g. message: (Jack, Jan 7th, 2022) jack went for hiking yesterday.
        You should update memory: jack went for hiking on Jan 6th, 2022.

Context:
- Speaker 0: {speaker_0}
- Speaker 1: {speaker_1}
"""


def create_evaluation_wrapper(
    config: Optional[M2AConfig] = None,
    db_dir: str = "eval_results",
) -> M2AEvaluationWrapper:
    """
    Create an M2A evaluation wrapper.

    Args:
        config: M2AConfig object. If None, loads from environment variables.
        db_dir: Directory to save evaluation results and intermediate data.

    Returns:
        M2AEvaluationWrapper instance
    """
    return M2AEvaluationWrapper(config=config, db_dir=db_dir)
