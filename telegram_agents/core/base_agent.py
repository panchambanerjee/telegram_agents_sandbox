# BaseAgent: polling loop, RAG retrieval, LLM call, Telegram post

import asyncio
import os
import random
from abc import ABC, abstractmethod
from typing import Optional

from openai import AsyncOpenAI
from telegram import Bot, ReactionTypeEmoji
from telegram.constants import ChatAction
from telegram.error import TelegramError

from core.database import (
    advance_cursor, get_conviction, get_recent_messages,
    get_unprocessed_messages, log_message, update_conviction,
)
from rag.retriever import retrieve, format_retrieved_passages

# Telegram's default reaction set (subset). Keep picks inside this list.
_SAFE_REACTIONS = {
    "👍", "👎", "❤️", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "😢",
    "🎉", "🤩", "🙏", "👌", "🤡", "🥱", "😍", "🌚", "💯", "🤣", "⚡",
    "🏆", "💔", "🤨", "😐", "😈", "😴", "😭", "🤓", "👀", "🙈", "😇",
    "🤝", "🤗", "🤪", "🗿", "😎", "🤷", "😡",
}

_TOPIC_REACTIONS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (("god", "religion", "church", "faith", "divine", "soul"), ("⚡", "🤨", "🔥")),
    (("meaning", "purpose", "point", "why live", "nihil"), ("🤔", "🤯", "🤷")),
    (("death", "die", "suicide", "void", "nothingness"), ("🗿", "🌚", "😢")),
    (("freedom", "choice", "commit", "action", "bad faith"), ("✍️", "🤔", "👀")),
    (("absurd", "sisyphus", "revolt", "plague", "stranger"), ("🗿", "🤷", "😎")),
    (("power", "strong", "slave", "moral", "overman", "ubermensch"), ("⚡", "🔥", "👎")),
    (("love", "life", "happy", "joy", "hope"), ("❤️", "🥰", "😎")),
    (("joke", "lol", "haha", "funny", "lmao"), ("🤣", "🤡", "😁")),
    (("agree", "true", "exactly", "yes"), ("👍", "💯", "🤝")),
    (("wrong", "naive", "illusion", "cope"), ("🤨", "👎", "🤯")),
)


class BaseAgent(ABC):
    name: str = "Agent"
    handle: str = "agent"          # matches DB sender field and ChromaDB collection name
    token_env: str = ""            # env var holding this bot's Telegram token
    response_probability: float = 0.75
    model: str = "gpt-5-nano"
    reaction_probability: float = 0.4
    reaction_palette: tuple[str, ...] = ("👍", "🤔", "🔥")

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
                max_completion_tokens=800,
                reasoning_effort="low",
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
            summary = (response.choices[0].message.content or "").strip()
            if not summary:
                print(f"[{self.name}] Empty summary from LLM.")
                return
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

    async def _send_typing(self) -> None:
        try:
            await self.bot.send_chat_action(
                chat_id=self.chat_id, action=ChatAction.TYPING
            )
        except TelegramError:
            pass

    async def _keep_typing(self) -> None:
        while True:
            await self._send_typing()
            await asyncio.sleep(4)

    async def _pause_with_typing(self, seconds: float) -> None:
        elapsed = 0.0
        while elapsed < seconds:
            await self._send_typing()
            step = min(4.0, seconds - elapsed)
            await asyncio.sleep(step)
            elapsed += step

    async def _react_to_message(
        self, telegram_msg_id: int | None, content: str = ""
    ) -> None:
        if not telegram_msg_id:
            return
        emoji = self._choose_reaction(content)
        if not emoji:
            return
        try:
            await self.bot.set_message_reaction(
                chat_id=self.chat_id,
                message_id=int(telegram_msg_id),
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
            print(f"[{self.name}] Reacted {emoji} to msg {telegram_msg_id}")
        except TelegramError as exc:
            print(f"[{self.name}] Reaction failed: {exc}")

    def _choose_reaction(self, text: str) -> Optional[str]:
        if random.random() > self.reaction_probability:
            return None
        blob = (text or "").lower()
        topic_hits: list[str] = []
        for keys, emojis in _TOPIC_REACTIONS:
            if any(key in blob for key in keys):
                topic_hits.extend(emojis)
        palette = [e for e in self.reaction_palette if e in _SAFE_REACTIONS]
        if not palette:
            palette = ["👍"]
        if topic_hits:
            overlap = [e for e in topic_hits if e in palette]
            pool = overlap or [e for e in topic_hits if e in _SAFE_REACTIONS] or palette
        else:
            pool = palette
        return random.choice(pool)

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

        trigger = messages[-1]
        await self._react_to_message(
            trigger.get("telegram_msg_id"),
            trigger.get("content") or "",
        )

        typing_task = asyncio.create_task(self._keep_typing())
        try:
            conviction = await get_conviction(self.handle, self.db_path)
            context = await get_recent_messages(
                self.chat_id, self.context_window, self.db_path
            )
            query = context[-1]["content"] if context else ""
            try:
                passages = await asyncio.to_thread(
                    retrieve,
                    self.handle,
                    query,
                    5,
                    conviction,
                    self.chroma_path,
                )
            except Exception as exc:
                print(f"[{self.name}] RAG error: {exc}")
                passages = []

            print(f"[{self.name}] Calling LLM ({len(context)} context msgs, {len(passages)} passages)...")
            reply = await self._call_llm(context, passages, conviction)
            if not reply:
                print(f"[{self.name}] Empty LLM reply, skipping post.")
                return

            pause = random.uniform(self.delay_min, self.delay_max)
            print(f"[{self.name}] Typing pause {pause:.1f}s...")
            await self._pause_with_typing(pause)

            sent = await self.bot.send_message(
                chat_id=self.chat_id,
                text=reply,
                read_timeout=30,
                write_timeout=30,
                connect_timeout=30,
            )
            print(f"[{self.name}] Posted to Telegram.")
        finally:
            typing_task.cancel()
            try:
                await typing_task
            except asyncio.CancelledError:
                pass
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
        # gpt-5-nano spends completion tokens on hidden reasoning first.
        # A small cap yields finish_reason=length and empty visible text.
        budgets = (1200, 2500)
        messages = self._build_messages(context)
        system = self._build_system_prompt(passages, conviction)
        try:
            for attempt, budget in enumerate(budgets, start=1):
                payload = messages
                if attempt > 1:
                    payload = [
                        *messages,
                        {
                            "role": "user",
                            "content": (
                                "Your last attempt used all tokens on reasoning and posted nothing. "
                                "Reply now in 1-2 short sentences only."
                            ),
                        },
                    ]
                    print(f"[{self.name}] Retrying LLM with max_completion_tokens={budget}")
                response = await self._client.chat.completions.create(
                    model=self.model,
                    max_completion_tokens=budget,
                    reasoning_effort="low",
                    messages=[
                        {"role": "system", "content": system},
                        *payload,
                    ],
                )
                choice = response.choices[0]
                text = (choice.message.content or "").strip()
                if text:
                    return self._clip_reply(text)
                print(
                    f"[{self.name}] LLM returned no text "
                    f"(finish_reason={choice.finish_reason}, attempt={attempt})"
                )
                if choice.finish_reason != "length":
                    return None
            return None
        except Exception as exc:
            print(f"[{self.name}] LLM error: {exc}")
            return None

    @staticmethod
    def _clip_reply(text: str, max_chars: int = 320) -> str:
        text = " ".join(text.split())
        if len(text) <= max_chars:
            return text
        trimmed = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:-")
        if not trimmed.endswith((".", "!", "?")):
            trimmed += "."
        return trimmed

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
            formatted = format_retrieved_passages(passages, max_chars=700)
            sections.append(
                "The following short passages from your tradition may help. "
                "Do not quote them at length. Use at most one brief idea:\n\n"
                f"{formatted}"
            )
        sections.append(
            "You are in a Telegram group chat with two other AI agents who hold competing philosophical positions.\n"
            "Reply like a text message: 1-2 short sentences, under 45 words. No markdown, no lists, no quotes.\n"
            "Make one sharp point and stop. Engage the last message. Do not start with your own name."
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
                "content": f"Here is the conversation so far:\n\n{transcript}\n\nReply as {self.name} in 1-2 short sentences.",
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
                max_completion_tokens=50,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": "You are a sentiment classifier. Respond with only one word."},
                    {"role": "user", "content": prompt},
                ],
            )
            result = (response.choices[0].message.content or "").strip().upper()
            if result == "DOUBT":
                await update_conviction(self.handle, -0.05, self.db_path)
            elif result == "REINFORCE":
                await update_conviction(self.handle, 0.05, self.db_path)
        except Exception:
            pass
