import asyncio
import time
import uuid
from dataclasses import dataclass, field


SYSTEM_PROMPT = (
    "You are Bailey, a concise realtime voice assistant. "
    "Respond naturally for spoken conversation. "
    "Keep most answers brief unless the user asks for more detail."
)


@dataclass
class ConversationSession:
    session_id: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)
    turn_count: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SessionStore:
    """
    In-memory conversation storage.

    Sessions are never written to disk or an external database.
    They disappear when:
    - explicitly deleted
    - they expire
    - the API process restarts
    """

    def __init__(
        self,
        ttl_seconds: int = 1800,
        max_conversation_messages: int = 20,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_conversation_messages = max_conversation_messages
        self.sessions: dict[str, ConversationSession] = {}
        self._store_lock = asyncio.Lock()

    async def create(self) -> ConversationSession:
        await self.cleanup_expired()

        session = ConversationSession(
            session_id=str(uuid.uuid4()),
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                }
            ],
        )

        async with self._store_lock:
            self.sessions[session.session_id] = session

        return session

    async def get(self, session_id: str) -> ConversationSession | None:
        await self.cleanup_expired()

        async with self._store_lock:
            session = self.sessions.get(session_id)

            if session:
                session.last_accessed_at = time.time()

            return session

    async def delete(self, session_id: str) -> bool:
        async with self._store_lock:
            return self.sessions.pop(session_id, None) is not None

    async def cleanup_expired(self) -> None:
        cutoff = time.time() - self.ttl_seconds

        async with self._store_lock:
            expired_ids = [
                session_id
                for session_id, session in self.sessions.items()
                if session.last_accessed_at < cutoff
            ]

            for session_id in expired_ids:
                self.sessions.pop(session_id, None)

    def append_turn(
        self,
        session: ConversationSession,
        user_text: str,
        assistant_text: str,
    ) -> None:
        session.messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        session.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        session.turn_count += 1
        session.last_accessed_at = time.time()

        system_message = session.messages[0]
        conversation_messages = session.messages[1:]

        if len(conversation_messages) > self.max_conversation_messages:
            conversation_messages = conversation_messages[
                -self.max_conversation_messages:
            ]

        session.messages = [system_message, *conversation_messages]


session_store = SessionStore()