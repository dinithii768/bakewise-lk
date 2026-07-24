"""
Base Agent class for BakeWise LK
All agents inherit from this class
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any
from agents.message import AgentMessage

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all BakeWise LK agents.
    Defines the standard interface for A2A communication.
    """

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"agent.{name}")

    def receive_message(self, message: AgentMessage) -> AgentMessage:
        """
        Entry point for receiving A2A messages.
        Routes to correct handler based on message_type.
        """
        self.logger.info(
            f"[{self.name}] received {message.message_type} "
            f"from {message.sender}"
        )

        try:
            if message.message_type == "task":
                return self.handle_task(message)
            elif message.message_type == "reflection_request":
                return self.handle_reflection(message)
            else:
                return self._error_response(
                    message,
                    f"Unknown message type: {message.message_type}"
                )
        except Exception as e:
            self.logger.error(f"[{self.name}] Error: {e}")
            return self._error_response(message, str(e))

    @abstractmethod
    def handle_task(self, message: AgentMessage) -> AgentMessage:
        pass

    def handle_reflection(self, message: AgentMessage) -> AgentMessage:
        return self._error_response(
            message,
            f"{self.name} does not support reflection"
        )

    def _reply(
        self,
        original: AgentMessage,
        content: Dict[str, Any]
    ) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=original.sender,
            message_type="result",
            content=content
        )

    def _error_response(
        self,
        original: AgentMessage,
        error: str
    ) -> AgentMessage:
        return AgentMessage(
            sender=self.name,
            receiver=original.sender,
            message_type="error",
            content={"error": error}
        )