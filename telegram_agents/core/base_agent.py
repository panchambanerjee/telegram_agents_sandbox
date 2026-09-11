# BaseAgent: polling loop, RAG retrieval, LLM call, Telegram post

import asyncio
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

from openai import AsyncOpenAI
from telegram import Bot
from telegram.error import TelegramError

from core.database import (
    advance_cursor, get_conviction, get_recent_messages,
    get_unprocessed_messages, log_message, update_conviction,
)
from rag.retriever import retrieve, format_retrieved_passages


class BaseAgent(ABC):
    name: str = "Agent"
    handle: str = "agent"          # matches DB sender field and ChromaDB collection name
    token_env: str = ""            # env var holding this bot's Telegram token
    response_probability: float = 0.75
    model: str = "gpt-5-nano"

    @property
    @abstractmethod
    def persona_prompt(self) -> str: ...

    def __init__(self) -> None:
        token = os.getenv(self.token_env)
        if not token:
            raise ValueError(f"Missing Telegram token in env var {self.token_env}")
        self.token = token
        self.bot = Bot(token=self.token)
        self.chat_id = int(os.getenv("GROUP_CHAT_ID", "0"))
        self.context_window = int(os.getenv("CONTEXT_WINDOW", "20"))
        self.delay_min = float(os.getenv("RESPONSE_DELAY_MIN", "3"))
        self.delay_max = float(os.getenv("RESPONSE_DELAY_MAX", "8"))
        self.db_path = os.getenv("DB_PATH", "./data/conversations.db")
        self.chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")
        self._client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self._running = False

    async def run(self) -> None:
        self._running = True
        print(f"[{self.name}] Started.")
        while self._running:
            try:
                await self._poll_and_respond()
            except Exception as exc:
                print(f"[{self.name}] Error: {exc}")
            await asyncio.sleep(2)

    async def stop(self) -> None:
        self._running = False

    async def post_summary(self) -> None:
        try:
            messages = await get_recent_messages(
                self.chat_id, limit=40, db_path=self.db_path
            )
            if not messages:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text="Nothing to summarise yet — drop a topic and let the philosophers argue.",
                )
                return

            transcript = "\n".join(
                f"{msg['sender_name']}: {msg['content']}" for msg in messages
            )
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=400,
                messages=[
                    {"role": "system", "content": self.persona_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Here is the philosophical debate so far:\n\n"
                            f"{transcript}\n\n"
                            "Summarise the key points of disagreement, any ground that has "
                            "shifted, and where each philosopher currently stands. Be concise. "
                            f"No markdown. Write as {self.name}."
                        ),
                    },
                ],
            )
            summary = response.choices[0].message.content.strip()
            await self.bot.send_message(chat_id=self.chat_id, text=summary)
            await log_message(
                chat_id=self.chat_id,
                sender=self.handle,
                sender_name=self.name,
                content=summary,
                db_path=self.db_path,
            )
        except Exception as exc:
            print(f"[{self.name}] Error in post_summary: {exc}")

    async def _poll_and_respond(self) -> None:
        messages = await get_unprocessed_messages(
            self.handle, self.chat_id, self.db_path
        )
        if not messages:
            return

        await advance_cursor(self.handle, messages[-1]["id"], self.db_path)

        if random.random() > self.response_probability:
            print(f"[{self.name}] Skipping this round.")
            return

        await asyncio.sleep(random.uniform(self.delay_min, self.delay_max))

        conviction = await get_conviction(self.handle, self.db_path)
        context = await get_recent_messages(
            self.chat_id, self.context_window, self.db_path
        )
        query = context[-1]["content"] if context else ""
        try:
            passages = retrieve(
                self.handle,
                query,
                conviction_score=conviction,
                chroma_path=self.chroma_path,
            )
        except Exception:
            passages = []

        reply = await self._call_llm(context, passages, conviction)
        if not reply:
            return

        sent = await self.bot.send_message(chat_id=self.chat_id, text=reply)
        await log_message(
            self.chat_id,
            self.handle,
            self.name,
            reply,
            telegram_msg_id=sent.message_id,
            db_path=self.db_path,
        )
        await self._update_conviction_from_reply(reply)

    async def _call_llm(self, context, passages, conviction) -> Optional[str]:
        try:
            messages = self._build_messages(context)
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=300,
                messages=[
                    {"role": "system", "content": self._build_system_prompt(passages, conviction)},
                    *messages,
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return None

    def _build_system_prompt(self, passages: list[dict], conviction: float) -> str:
        if conviction < 0.35:
            conviction_state = (
                "Your certainty in your position is weakening. You are beginning to "
                "question your core assumptions, though you have not abandoned them."
            )
        elif conviction > 0.65:
            conviction_state = (
                "Your conviction is strong. You are doubling down on your position "
                "and finding it increasingly coherent."
            )
        else:
            conviction_state = (
                "You hold your position with moderate confidence. You are open to "
                "strong arguments but not easily moved."
            )

        sections = [self.persona_prompt, conviction_state]
        if passages:
            formatted = format_retrieved_passages(passages)
            sections.append(
                "The following passages from your philosophical tradition are relevant "
                "to this conversation. Draw on them to ground your arguments:\n\n"
                f"{formatted}"
            )
        sections.append(
            "You are in a Telegram group chat with two other AI agents who hold competing philosophical positions.\n"
            "Keep replies to 2-4 sentences. Write like you are texting — no markdown, no bullet points.\n"
            "Engage directly with what was just said. Do not start your message with your own name."
        )
        return "\n\n".join(sections)

    def _build_messages(self, context: list[dict]) -> list[dict]:
        if not context:
            return [
                {
                    "role": "user",
                    "content": "The conversation is just starting. Introduce your philosophical position briefly.",
                }
            ]
        transcript = "\n".join(
            f"{msg['sender_name']}: {msg['content']}" for msg in context
        )
        return [
            {
                "role": "user",
                "content": f"Here is the conversation so far:\n\n{transcript}\n\nReply as {self.name}.",
            }
        ]

    async def _update_conviction_from_reply(self, reply: str) -> None:
        try:
            prompt = (
                "Does the following statement express doubt, uncertainty, or "
                "weakening of a philosophical position, or does it express "
                "confidence and reinforcement?\n"
                "Reply with exactly one word: DOUBT or REINFORCE.\n\n"
                f"Statement: {reply}"
            )
            response = await self._client.chat.completions.create(
                model=self.model,
                max_tokens=5,
                messages=[
                    {"role": "system", "content": "You are a sentiment classifier. Respond with only one word."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = response.choices[0].message.content.strip().upper()
            if result == "DOUBT":
                await update_conviction(self.handle, -0.05, self.db_path)
            elif result == "REINFORCE":
                await update_conviction(self.handle, 0.05, self.db_path)
        except Exception:
            pass
