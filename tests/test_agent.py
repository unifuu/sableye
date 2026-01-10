import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent import SableyeAgent
from src.config import Config


@pytest.fixture
def mock_config(temp_vault_dir):
    """Create a mock configuration"""
    config = Config(config_path=None)
    config.model.type = "ollama"
    config.model.name = "qwen3:4b-instruct"
    config.vault.path = str(temp_vault_dir)
    return config


@pytest.fixture
def mock_llm():
    """Create a mock LLM"""
    llm = Mock()
    llm.invoke.return_value = MagicMock(content="Test response")
    return llm


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model"""
    embeddings = Mock()
    return embeddings


class TestSableyeAgent:
    """Test cases for SableyeAgent class"""
    
    def test_initialization(self, mock_config):
        """Test agent initialization"""
        with patch('src.agent.ChatOllama') as mock_chat_ollama, \
             patch('src.agent.OllamaEmbeddings') as mock_ollama_embeddings:
            
            mock_chat_ollama.return_value = Mock()
            mock_ollama_embeddings.return_value = Mock()
            
            agent = SableyeAgent(mock_config)
            
            assert agent.config == mock_config
            assert agent.reader is None
            assert agent.vectorstore is None
            assert agent.agent is None
            assert agent.chat_history == []
            assert agent.memory is not None
    
    def test_initialize_models_openai(self, monkeypatch):
        """Test model initialization with OpenAI"""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        
        config = Config(config_path=None)
        config.model.type = "openai"
        config.vault.path = "/tmp"
        
        with patch('src.agent.ChatOpenAI') as mock_chat_openai, \
             patch('src.agent.OpenAIEmbeddings') as mock_openai_embeddings:
            
            mock_chat_openai.return_value = Mock()
            mock_openai_embeddings.return_value = Mock()
            
            agent = SableyeAgent(config)
            
            assert agent.llm is not None
            assert agent.embeddings is not None
            mock_chat_openai.assert_called_once()
            mock_openai_embeddings.assert_called_once()
    
    def test_initialize_models_ollama(self, mock_config):
        """Test model initialization with Ollama"""
        with patch('src.agent.ChatOllama') as mock_chat_ollama, \
             patch('src.agent.OllamaEmbeddings') as mock_ollama_embeddings:
            
            mock_chat_ollama.return_value = Mock()
            mock_ollama_embeddings.return_value = Mock()
            
            agent = SableyeAgent(mock_config)
            
            assert agent.llm is not None
            assert agent.embeddings is not None
            mock_chat_ollama.assert_called_once()
            mock_ollama_embeddings.assert_called_once()
    
    def test_initialize_memory(self, mock_config):
        """Test memory initialization"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            
            assert agent.memory is not None
            assert agent.memory.memory_key == "chat_history"
            assert agent.memory.return_messages is True
    
    def test_load_notes(self, mock_config):
        """Test loading notes into vector store"""
        with patch('src.agent.ChatOllama'), \
             patch('src.agent.OllamaEmbeddings') as mock_embeddings, \
             patch('src.agent.FAISS') as mock_faiss:
            
            mock_faiss.from_documents.return_value = Mock()
            
            agent = SableyeAgent(mock_config)
            agent.load_notes(days=30)
            
            assert agent.reader is not None
            assert agent.vectorstore is not None
            mock_faiss.from_documents.assert_called_once()
    
    def test_load_notes_no_documents(self, mock_config):
        """Test load_notes raises error when no documents found"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            
            # Create empty vault directory
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                agent.config.vault.path = tmpdir
                with pytest.raises(ValueError, match="No documents found"):
                    agent.load_notes()
    
    def test_initialize(self, mock_config):
        """Test agent initialization"""
        with patch('src.agent.ChatOllama'), \
             patch('src.agent.OllamaEmbeddings') as mock_embeddings, \
             patch('src.agent.FAISS') as mock_faiss, \
             patch('src.agent.create_tool_calling_agent') as mock_create_agent, \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_faiss.from_documents.return_value = Mock()
            mock_create_agent.return_value = Mock()
            mock_executor.return_value = Mock()
            
            agent = SableyeAgent(mock_config)
            agent.load_notes()
            agent.initialize()
            
            assert agent.agent_executor is not None
            mock_create_agent.assert_called_once()
            mock_executor.assert_called_once()
    
    def test_initialize_without_loading_notes(self, mock_config):
        """Test initialize raises error without loading notes first"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            
            with pytest.raises(ValueError, match="Please load notes first"):
                agent.initialize()
    
    def test_chat(self, mock_config):
        """Test chat functionality"""
        with patch('src.agent.ChatOllama'), \
             patch('src.agent.OllamaEmbeddings') as mock_embeddings, \
             patch('src.agent.FAISS') as mock_faiss, \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_faiss.from_documents.return_value = Mock()
            mock_executor.return_value = Mock()
            mock_executor.return_value.invoke.return_value = {"output": "Test response"}
            
            agent = SableyeAgent(mock_config)
            agent.load_notes()
            agent.initialize()
            
            response = agent.chat("Hello")
            
            assert response == "Test response"
            mock_executor.return_value.invoke.assert_called_once_with({"input": "Hello"})
    
    def test_chat_not_initialized(self, mock_config):
        """Test chat raises error when agent not initialized"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            agent.agent_executor = None  # Ensure attribute exists
            
            with pytest.raises(ValueError, match="Agent not initialized"):
                agent.chat("Hello")
    
    def test_chat_error_handling(self, mock_config):
        """Test chat error handling"""
        with patch('src.agent.ChatOllama'), \
             patch('src.agent.OllamaEmbeddings') as mock_embeddings, \
             patch('src.agent.FAISS') as mock_faiss, \
             patch('src.agent.create_tool_calling_agent'), \
             patch('src.agent.AgentExecutor') as mock_executor:
            
            mock_faiss.from_documents.return_value = Mock()
            mock_executor.return_value = Mock()
            mock_executor.return_value.invoke.side_effect = Exception("Chat error")
            
            agent = SableyeAgent(mock_config)
            agent.load_notes()
            agent.initialize()
            
            response = agent.chat("Hello")
            
            assert "I encountered an error" in response
    
    def test_clear_memory(self, mock_config):
        """Test clearing conversation memory"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            agent.memory.chat_memory.add_user_message("Test")
            
            agent.clear_memory()
            
            assert len(agent.memory.chat_memory.messages) == 0
            assert agent.chat_history == []
    
    def test_get_memory_summary(self, mock_config):
        """Test getting memory summary"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            agent.memory.chat_memory.add_user_message("Test message")
            
            summary = agent.get_memory_summary()
            
            assert "Conversation history" in summary
            assert "User: Test message" in summary
    
    def test_get_memory_summary_empty(self, mock_config):
        """Test memory summary with empty history"""
        with patch('src.agent.ChatOllama'), patch('src.agent.OllamaEmbeddings'):
            agent = SableyeAgent(mock_config)
            
            summary = agent.get_memory_summary()
            
            assert summary == "No conversation history"
