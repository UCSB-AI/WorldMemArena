from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI

from .config import M2AConfig, get_default_config
from .stores import ImageManager, RawMessageStore, SemanticStore
from .agents.memory_manager import MemoryManagerTools, MemoryManager
from .agents.chat_agent import ChatAgent


class M2ASystem:
    def __init__(self, config: Optional[M2AConfig] = None):
        """
        Initialize M2A system with optional configuration.

        Args:
            config: M2AConfig object. If None, loads from environment variables.
        """
        self.config = config or get_default_config()

        # Initialize storage layers with configuration
        self.raw_store = RawMessageStore(
            db_path=self.config.memory.raw_db_path,
            reuse=self.config.memory.reuse_db
        )

        self.semantic_store = SemanticStore(
            db_path=self.config.memory.semantic_db_path,
            re_use=self.config.memory.reuse_db,
            text_embedder_config=self.config.text_embedding,
            multimodal_embedder_config=self.config.multimodal_embedding
        )

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.config.llm.model,
            base_url=self.config.llm.base_url,
            api_key=self.config.llm.api_key,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            timeout=self.config.llm.timeout,
        )

        # Initialize ImageManager with format configuration
        image_manager = ImageManager()
        self.image_manager = image_manager
        # Initialize MemoryManager
        mm_tools = MemoryManagerTools(
            self.raw_store,
            self.semantic_store,
            image_manager,
            max_raw_msg=self.config.memory.max_raw_messages_return,
        )
        self.memory_manager = MemoryManager(
            tools=mm_tools,
            raw_store=self.raw_store,
            semantic_store=self.semantic_store,
            llm=self.llm,
            image_manager=image_manager,
            config=self.config.memory_manager
        )

        # Initialize ChatAgent
        self.chat_agent = ChatAgent(
            memory_manager=self.memory_manager,
            raw_store=self.raw_store,
            llm=self.llm,
            image_manager=image_manager,
            update_memory=True,
            config=config.chat_agent
        )

    def chat_once(
        self,
        text: str,
        image: Optional[str] = None,
        timestamp: Optional[str] = None,
        role: Optional[str] = None
    ) -> str:
        """
        User-facing chat interface for single message.

        Args:
            text: The text message
            image: Optional image file path
            timestamp: Optional timestamp for the message (for evaluation)
            role: Optional role name (for evaluation)

        Returns:
            The assistant's response
        """
        return self.chat_agent.chat(
            user_text=text,
            user_image_path_or_url=image,
            timestamp=timestamp,
            role=role
        )

    def set_system_prompt(self, system_prompt: str):
        """Set a custom system prompt for the chat agent"""
        self.chat_agent.system_prompt = system_prompt



def create_m2a_from_env() -> M2ASystem:
    """Create M2A system loading configuration from environment variables"""
    return M2ASystem(config=M2AConfig.from_env())


def create_m2a_from_file(config_path: str) -> M2ASystem:
    """Create M2A system loading configuration from file"""
    return M2ASystem(config=M2AConfig.from_file(config_path))
