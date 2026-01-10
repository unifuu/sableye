import logging
from typing import Optional
from langchain.agents import create_agent
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from .reader import ObsidianReader
from .tools import AgentTools
from .config import Config

logger = logging.getLogger(__name__)


class SableyeAgent:
    """AI agent for analyzing mental health and goals from Obsidian notes"""
    
    def __init__(self, config: Config):
        self.config = config
        self.reader = None
        self.vectorstore = None
        self.agent = None
        self.llm = None
        self.embeddings = None
        self.chat_history = []
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize LLM and embeddings"""
        logger.info(f"Initializing {self.config.model.type} model: {self.config.model.name}")
        
        if self.config.model.type == "openai":
            self.llm = ChatOpenAI(
                model=self.config.model.name,
                api_key=self.config.model.api_key,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens
            )
            self.embeddings = OpenAIEmbeddings(api_key=self.config.model.api_key)
        
        elif self.config.model.type == "ollama":
            self.llm = ChatOllama(
                model=self.config.model.name,
                base_url=self.config.model.base_url,
                temperature=self.config.model.temperature
            )
            # Use a dedicated embedding model for Ollama
            self.embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=self.config.model.base_url
            )
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model.type}")
    
    def load_notes(self, days: Optional[int] = None):
        """Load notes into vector store"""
        logger.info("Loading notes from Obsidian vault...")
        
        self.reader = ObsidianReader(
            vault_path=self.config.vault.path,
            chunk_size=self.config.vault.chunk_size,
            chunk_overlap=self.config.vault.chunk_overlap
        )
        
        # Load documents
        load_days = days or self.config.vault.load_days
        if load_days:
            documents = self.reader.read_recent_notes(load_days)
        else:
            documents = self.reader.read_all_notes()
        
        if not documents:
            raise ValueError("No documents found in vault")
        
        # Split documents
        splits = self.reader.split_documents(documents)
        
        # Create vector store
        logger.info("Creating vector store...")
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)
        
        logger.info(
            f"Loaded {len(documents)} documents ({len(splits)} chunks) "
            f"from last {load_days or 'all'} days"
        )
    
    def initialize(self):
        """Initialize the agent with tools"""
        if not self.vectorstore:
            raise ValueError("Please load notes first using load_notes()")
        
        logger.info("Initializing agent...")
        
        # Create tools
        agent_tools = AgentTools(
            vectorstore=self.vectorstore,
            reader=self.reader,
            search_limit=self.config.agent.search_results_limit
        )
        tools = agent_tools.create_tools()
        
        # Create prompt with memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent using the new API
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=self._get_system_prompt()
        )
        
        logger.info("Agent initialized successfully")
    
    def chat(self, message: str) -> str:
        """Chat with the agent"""
        if not self.agent:
            raise ValueError("Agent not initialized. Call initialize() first.")
        
        try:
            # Add message to chat history
            self.chat_history.append({"role": "user", "content": message})
            
            # Create messages list with history
            messages = []
            for msg in self.chat_history[-10:]:  # Keep last 10 messages
                messages.append(msg)
            
            response = self.agent.invoke({"messages": messages})
            
            # Extract the last message content from the response
            if "messages" in response and response["messages"]:
                last_message = response["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                elif isinstance(last_message, dict) and 'content' in last_message:
                    response_content = last_message['content']
                else:
                    response_content = str(last_message)
            else:
                response_content = str(response)
            
            # Add response to chat history
            self.chat_history.append({"role": "assistant", "content": response_content})
            
            return response_content
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"I encountered an error: {str(e)}"
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.chat_history = []
        logger.info("Conversation memory cleared")
    
    def get_memory_summary(self) -> str:
        """Get a summary of conversation history"""
        if not self.chat_history:
            return "No conversation history"
        
        summary = f"Conversation history ({len(self.chat_history)} messages):\n"
        for msg in self.chat_history[-10:]:  # Last 10 messages
            role = "User" if msg["role"] == "user" else "Agent"
            content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            summary += f"- {role}: {content}\n"
        
        return summary
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent"""
        return ""