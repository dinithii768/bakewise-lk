"""
Agent Message Protocol for BakeWise LK
A2A (Agent-to-Agent) structured communication
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass
class AgentMessage:
    """
    Structured message passed between agents.
    Implements A2A communication protocol.
    """
    sender: str
    receiver: str
    message_type: str  # task | result | reflection_request | error
    content: Dict[str, Any]
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    message_id: str = field(
        default_factory=lambda: f"msg_{datetime.now().timestamp():.0f}"
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type,
            "content": self.content,
            "timestamp": self.timestamp,
            "message_id": self.message_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentMessage":
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            message_type=data["message_type"],
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            message_id=data.get("message_id", f"msg_{datetime.now().timestamp():.0f}")
        )

    def __repr__(self):
        return (
            f"AgentMessage("
            f"sender={self.sender}, "
            f"receiver={self.receiver}, "
            f"type={self.message_type}, "
            f"id={self.message_id})"
        )