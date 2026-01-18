"""
Session management for the NanoGPT Chat Server.

Handles session creation, retrieval, updates, and cleanup.
"""

import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple

from .config import ChatConfig


class SessionManager:
    """Manages chat sessions and conversation histories."""

    def __init__(self, timeout_minutes: int = ChatConfig.SESSION_TIMEOUT_MINUTES):
        """
        Initialize the session manager.

        Args:
            timeout_minutes: Number of minutes before a session expires.
        """
        self.sessions: Dict[str, Dict] = {}
        self.lock = Lock()
        self.timeout_minutes = timeout_minutes

    def create_session(self) -> str:
        """
        Create a new session and return the session ID.

        Returns:
            The new session ID.
        """
        session_id = str(uuid.uuid4())
        with self.lock:
            self.sessions[session_id] = {
                "conversation": [],
                "last_accessed": datetime.now(),
            }
        return session_id

    def get_session(self, session_id: Optional[str]) -> Tuple[str, List[dict]]:
        """
        Get or create a session. Returns (session_id, conversation_history).

        If session_id is None or invalid, creates a new session.
        Updates last_accessed timestamp.

        Args:
            session_id: The session ID, or None to create a new session.

        Returns:
            A tuple of (session_id, conversation_history).
        """
        with self.lock:
            # Clean up expired sessions
            self._cleanup_expired_sessions()

            # If no session_id provided or session doesn't exist, create new one
            if not session_id or session_id not in self.sessions:
                session_id = str(uuid.uuid4())
                self.sessions[session_id] = {
                    "conversation": [],
                    "last_accessed": datetime.now(),
                }
            else:
                # Update last accessed time
                self.sessions[session_id]["last_accessed"] = datetime.now()

            return session_id, self.sessions[session_id]["conversation"]

    def update_conversation(self, session_id: str, conversation: List[dict]):
        """
        Update the conversation history for a session.

        Args:
            session_id: The session ID.
            conversation: The updated conversation history.
        """
        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["conversation"] = conversation
                self.sessions[session_id]["last_accessed"] = datetime.now()

    def clear_conversation(self, session_id: Optional[str]) -> bool:
        """
        Clear conversation history for a session.

        Args:
            session_id: The session ID.

        Returns:
            True if session existed and was cleared, False otherwise.
        """
        if not session_id:
            return False

        with self.lock:
            if session_id in self.sessions:
                self.sessions[session_id]["conversation"] = []
                self.sessions[session_id]["last_accessed"] = datetime.now()
                return True
        return False

    def session_exists(self, session_id: str) -> bool:
        """
        Check if a session exists.

        Args:
            session_id: The session ID to check.

        Returns:
            True if the session exists, False otherwise.
        """
        with self.lock:
            return session_id in self.sessions

    def get_session_data(self, session_id: str) -> Optional[dict]:
        """
        Get full session data including metadata.

        Args:
            session_id: The session ID.

        Returns:
            The session data dict, or None if session doesn't exist.
        """
        with self.lock:
            if session_id not in self.sessions:
                return None

            session_data = self.sessions[session_id]
            return {
                "session_id": session_id,
                "conversation_length": len(session_data["conversation"]),
                "last_accessed": session_data["last_accessed"],
                "conversation": session_data["conversation"],
            }

    def get_statistics(self) -> dict:
        """
        Get statistics about active sessions.

        Returns:
            A dict containing statistics.
        """
        with self.lock:
            active_sessions = len(self.sessions)
            total_conversations = sum(
                len(s["conversation"]) for s in self.sessions.values()
            )

            return {
                "active_sessions": active_sessions,
                "total_conversation_messages": total_conversations,
            }

    def _cleanup_expired_sessions(self):
        """Remove sessions that haven't been accessed in timeout_minutes."""
        current_time = datetime.now()
        expired_sessions = [
            sid
            for sid, data in self.sessions.items()
            if (current_time - data["last_accessed"]).total_seconds()
            > self.timeout_minutes * 60
        ]
        for sid in expired_sessions:
            del self.sessions[sid]

        if expired_sessions:
            print(f"🧹 Cleaned up {len(expired_sessions)} expired session(s)")
