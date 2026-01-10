import logging
from typing import List, Callable
from langchain_core.tools import Tool
from langchain_community.vectorstores import FAISS

logger = logging.getLogger(__name__)


class AgentTools:
    """Collection of tools for the mental health agent"""
    
    def __init__(self, vectorstore: FAISS, reader, search_limit: int = 5):
        self.vectorstore = vectorstore
        self.reader = reader
        self.search_limit = search_limit
    
    def create_tools(self) -> List[Tool]:
        """Create all tools for the agent"""
        return [
            Tool(
                name="search_notes",
                func=self._search_notes,
                description=(
                    "Search through journal entries and notes for specific information. "
                    "Input should be a search query about goals, activities, thoughts, "
                    "emotions, or any topic you want to find in the notes."
                )
            ),
            Tool(
                name="get_recent_entries",
                func=self._get_recent_entries,
                description=(
                    "Get recent journal entries from the last N days. "
                    "Input should be a number representing days (e.g., '7' for last 7 days, "
                    "'30' for last 30 days)."
                )
            ),
            Tool(
                name="analyze_mood_patterns",
                func=self._analyze_mood_patterns,
                description=(
                    "Analyze mood and emotional patterns from journal entries. "
                    "Use this to understand mental state, emotional well-being, "
                    "and identify trends in emotions over time."
                )
            ),
            Tool(
                name="find_goals",
                func=self._find_goals,
                description=(
                    "Search for goals, objectives, and aspirations mentioned in notes. "
                    "Use this to understand what the person is working towards or wants to achieve."
                )
            )
        ]
    
    def _search_notes(self, query: str) -> str:
        """Search through journal entries and notes"""
        try:
            docs = self.vectorstore.similarity_search(query, k=self.search_limit)
            
            if not docs:
                return "No relevant entries found for this query."
            
            results = []
            for i, doc in enumerate(docs, 1):
                file_name = doc.metadata.get('file_name', 'Unknown')
                modified = doc.metadata.get('modified_time', 'Unknown')
                
                results.append(
                    f"--- Entry {i}: {file_name} (Modified: {modified[:10]}) ---\n"
                    f"{doc.page_content}\n"
                )
            
            return "\n".join(results)
        
        except Exception as e:
            logger.error(f"Error in search_notes: {e}")
            return f"Error searching notes: {str(e)}"
    
    def _get_recent_entries(self, days: str = "7") -> str:
        """Get recent journal entries"""
        try:
            n_days = int(days)
        except ValueError:
            n_days = 7
        
        try:
            recent_docs = self.reader.read_recent_notes(n_days)
            
            if not recent_docs:
                return f"No entries found in the last {n_days} days."
            
            # Sort by modified time
            recent_docs.sort(
                key=lambda x: x.metadata.get('modified_time', ''),
                reverse=True
            )
            
            results = []
            for doc in recent_docs[:10]:  # Limit to 10 most recent
                file_name = doc.metadata.get('file_name', 'Unknown')
                modified = doc.metadata.get('modified_time', 'Unknown')
                content = doc.page_content[:500]  # First 500 chars
                
                results.append(
                    f"--- {file_name} (Modified: {modified[:10]}) ---\n"
                    f"{content}{'...' if len(doc.page_content) > 500 else ''}\n"
                )
            
            return "\n".join(results)
        
        except Exception as e:
            logger.error(f"Error in get_recent_entries: {e}")
            return f"Error retrieving recent entries: {str(e)}"
    
    def _analyze_mood_patterns(self, query: str = "") -> str:
        """Analyze mood patterns"""
        try:
            # Search for mood-related content
            mood_keywords = [
                "mood", "emotion", "feeling", "mental state",
                "happiness", "sadness", "anxiety", "stress",
                "joy", "depression", "calm", "peaceful"
            ]
            
            mood_query = " ".join(mood_keywords)
            docs = self.vectorstore.similarity_search(mood_query, k=10)
            
            if not docs:
                return "No mood-related entries found."
            
            entries = []
            for doc in docs:
                file_name = doc.metadata.get('file_name', 'Unknown')
                entries.append(f"--- {file_name} ---\n{doc.page_content}\n")
            
            return "Mood-related entries:\n\n" + "\n".join(entries)
        
        except Exception as e:
            logger.error(f"Error in analyze_mood_patterns: {e}")
            return f"Error analyzing mood patterns: {str(e)}"
    
    def _find_goals(self, query: str = "") -> str:
        """Find goals and objectives"""
        try:
            goal_keywords = [
                "goal", "objective", "want to", "plan to",
                "aspire", "achieve", "target", "aim",
                "working on", "working towards"
            ]
            
            goal_query = " ".join(goal_keywords)
            docs = self.vectorstore.similarity_search(goal_query, k=8)
            
            if not docs:
                return "No goals found in the notes."
            
            entries = []
            for doc in docs:
                file_name = doc.metadata.get('file_name', 'Unknown')
                entries.append(f"--- {file_name} ---\n{doc.page_content}\n")
            
            return "Goal-related entries:\n\n" + "\n".join(entries)
        
        except Exception as e:
            logger.error(f"Error in find_goals: {e}")
            return f"Error finding goals: {str(e)}"