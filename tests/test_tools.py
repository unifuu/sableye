import pytest
from unittest.mock import Mock, MagicMock, patch
from src.tools import AgentTools
from langchain_core.documents import Document


@pytest.fixture
def mock_vectorstore():
    """Create a mock FAISS vectorstore"""
    vectorstore = Mock()
    
    # Mock similarity search to return test documents
    test_docs = [
        Document(
            page_content="I feel happy and energetic today.",
            metadata={"file_name": "test1.md", "modified_time": "2024-01-01T10:00:00"}
        ),
        Document(
            page_content="I'm feeling a bit anxious about the upcoming deadline.",
            metadata={"file_name": "test2.md", "modified_time": "2024-01-02T11:00:00"}
        )
    ]
    vectorstore.similarity_search.return_value = test_docs
    
    return vectorstore


@pytest.fixture
def mock_reader():
    """Create a mock ObsidianReader"""
    reader = Mock()
    
    test_docs = [
        Document(
            page_content="Recent entry about goals",
            metadata={"file_name": "recent1.md", "modified_time": "2024-01-10T10:00:00"}
        ),
        Document(
            page_content="Another recent entry",
            metadata={"file_name": "recent2.md", "modified_time": "2024-01-09T10:00:00"}
        )
    ]
    reader.read_recent_notes.return_value = test_docs
    
    return reader


@pytest.fixture
def agent_tools(mock_vectorstore, mock_reader):
    """Create AgentTools instance with mocked dependencies"""
    return AgentTools(
        vectorstore=mock_vectorstore,
        reader=mock_reader,
        search_limit=5
    )


class TestAgentTools:
    """Test cases for AgentTools class"""
    
    def test_initialization(self, mock_vectorstore, mock_reader):
        """Test AgentTools initialization"""
        tools = AgentTools(
            vectorstore=mock_vectorstore,
            reader=mock_reader,
            search_limit=10
        )
        
        assert tools.vectorstore == mock_vectorstore
        assert tools.reader == mock_reader
        assert tools.search_limit == 10
    
    def test_create_tools(self, agent_tools):
        """Test tool creation"""
        tools = agent_tools.create_tools()
        
        assert len(tools) == 4
        tool_names = [tool.name for tool in tools]
        assert "search_notes" in tool_names
        assert "get_recent_entries" in tool_names
        assert "analyze_mood_patterns" in tool_names
        assert "find_goals" in tool_names
    
    def test_search_notes(self, agent_tools, mock_vectorstore):
        """Test search_notes tool"""
        result = agent_tools._search_notes("happy")
        
        # Verify vectorstore was called
        mock_vectorstore.similarity_search.assert_called_once_with("happy", k=5)
        
        # Verify result format
        assert "Entry 1:" in result
        assert "test1.md" in result
        assert "I feel happy and energetic today" in result
    
    def test_search_notes_empty_results(self, mock_vectorstore, mock_reader):
        """Test search_notes with no results"""
        mock_vectorstore.similarity_search.return_value = []
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._search_notes("nonexistent")
        
        assert "No relevant entries found" in result
    
    def test_search_notes_error_handling(self, mock_vectorstore, mock_reader):
        """Test search_notes error handling"""
        mock_vectorstore.similarity_search.side_effect = Exception("Search error")
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._search_notes("test")
        
        assert "Error searching notes" in result
    
    def test_get_recent_entries(self, agent_tools, mock_reader):
        """Test get_recent_entries tool"""
        result = agent_tools._get_recent_entries("7")
        
        # Verify reader was called
        mock_reader.read_recent_notes.assert_called_once_with(7)
        
        # Verify result format
        assert "recent1.md" in result
        assert "Recent entry about goals" in result
    
    def test_get_recent_entries_invalid_days(self, agent_tools, mock_reader):
        """Test get_recent_entries with invalid days parameter"""
        result = agent_tools._get_recent_entries("invalid")
        
        # Should default to 7 days
        mock_reader.read_recent_notes.assert_called_once_with(7)
    
    def test_get_recent_entries_no_results(self, mock_vectorstore, mock_reader):
        """Test get_recent_entries with no results"""
        mock_reader.read_recent_notes.return_value = []
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._get_recent_entries("7")
        
        assert "No entries found" in result
    
    def test_get_recent_entries_error_handling(self, mock_vectorstore, mock_reader):
        """Test get_recent_entries error handling"""
        mock_reader.read_recent_notes.side_effect = Exception("Read error")
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._get_recent_entries("7")
        
        assert "Error retrieving recent entries" in result
    
    def test_analyze_mood_patterns(self, agent_tools, mock_vectorstore):
        """Test analyze_mood_patterns tool"""
        result = agent_tools._analyze_mood_patterns()
        
        # Verify vectorstore was called with mood keywords
        mock_vectorstore.similarity_search.assert_called_once()
        call_args = mock_vectorstore.similarity_search.call_args
        query = call_args[0][0]
        
        # Check that mood keywords are in query
        assert "mood" in query
        assert "emotion" in query
        
        # Verify result format
        assert "Mood-related entries:" in result
    
    def test_analyze_mood_patterns_no_results(self, mock_vectorstore, mock_reader):
        """Test analyze_mood_patterns with no results"""
        mock_vectorstore.similarity_search.return_value = []
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._analyze_mood_patterns()
        
        assert "No mood-related entries found" in result
    
    def test_find_goals(self, agent_tools, mock_vectorstore):
        """Test find_goals tool"""
        result = agent_tools._find_goals()
        
        # Verify vectorstore was called with goal keywords
        mock_vectorstore.similarity_search.assert_called_once()
        call_args = mock_vectorstore.similarity_search.call_args
        query = call_args[0][0]
        
        # Check that goal keywords are in query
        assert "goal" in query
        assert "objective" in query
        
        # Verify result format
        assert "Goal-related entries:" in result
    
    def test_find_goals_no_results(self, mock_vectorstore, mock_reader):
        """Test find_goals with no results"""
        mock_vectorstore.similarity_search.return_value = []
        
        tools = AgentTools(mock_vectorstore, mock_reader, 5)
        result = tools._find_goals()
        
        assert "No goals found" in result
    
    def test_tool_descriptions(self, agent_tools):
        """Test that tools have proper descriptions"""
        tools = agent_tools.create_tools()
        
        for tool in tools:
            assert tool.description
            assert len(tool.description) > 10
