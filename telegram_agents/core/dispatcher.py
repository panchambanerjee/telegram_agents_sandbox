# Dispatcher: receives Telegram updates, logs human messages, routes commands

import os

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from core.database import (
    clear_chat,
    get_all_convictions,
    log_message,
    message_already_logged,
)

_BOT_NAMES = {"nietzsche", "sartre", "camus"}
_DISPLAY_NAMES = {
    "nihilist": "Nietzsche",
    "existentialist": "Sartre",
    "absurdist": "Camus",
}
_DISPLAY_ORDER = ("nihilist", "existentialist", "absurdist")


class Dispatcher:
    def __init__(self, agents: dict) -> None:
        self.agents = agents
        self.chat_id = int(os.getenv("GROUP_CHAT_ID", "0"))
        self.db_path = os.getenv("DB_PATH", "./data/conversations.db")
        listener_token = os.getenv("NIHILIST_BOT_TOKEN", "")
        self.app = Application.builder().token(listener_token).build()
        self._warned_chats: set[int] = set()
        self._register_handlers()
        if self.chat_id in (0, -1001234567890):
            print(
                f"[Dispatcher] GROUP_CHAT_ID={self.chat_id} looks like a placeholder. "
                "Send /help in your Telegram group, copy the chat id from the bot reply "
                "into telegram_agents/.env, and restart."
            )

    def _wrong_chat(self, update: Update) -> bool:
        msg = update.effective_message
        if msg is None:
            return True
        if msg.chat_id == self.chat_id:
            return False
        if msg.chat_id not in self._warned_chats:
            self._warned_chats.add(msg.chat_id)
            print(
                f"[Dispatcher] Saw chat_id={msg.chat_id} but GROUP_CHAT_ID={self.chat_id}. "
                f"Set GROUP_CHAT_ID={msg.chat_id} in .env and restart."
            )
        return True

    def _register_handlers(self) -> None:
        self.app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_message)
        )
        self.app.add_handler(CommandHandler("reset", self._on_reset))
        self.app.add_handler(CommandHandler("summary", self._on_summary))
        self.app.add_handler(CommandHandler("convictions", self._on_convictions))
        self.app.add_handler(CommandHandler("help", self._on_help))

    async def _on_message(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        msg = update.message
        if msg is None or self._wrong_chat(update):
            return

        user = msg.from_user
        if user is not None:
            first = (user.first_name or "").lower()
            username = (user.username or "").lower()
            if first in _BOT_NAMES or username in _BOT_NAMES:
                return

        if await message_already_logged(msg.message_id, self.db_path):
            return

        sender_name = "Human"
        if user is not None:
            sender_name = user.first_name or user.username or "Human"

        await log_message(
            chat_id=self.chat_id,
            sender="user",
            sender_name=sender_name,
            content=msg.text,
            telegram_msg_id=msg.message_id,
            db_path=self.db_path,
        )
        print(f"[Dispatcher] Logged from {sender_name}: {msg.text[:60]}")

    async def _on_reset(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or self._wrong_chat(update):
            return
        await clear_chat(self.chat_id, self.db_path)
        await update.message.reply_text(
            "🔄 Conversation reset. Drop a new topic and the philosophers will engage."
        )

    async def _on_summary(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or self._wrong_chat(update):
            return
        camus = self.agents.get("absurdist")
        if camus is not None:
            await camus.post_summary()
        else:
            await update.message.reply_text("Absurdist agent not available.")

    async def _on_convictions(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or self._wrong_chat(update):
            return
        scores = await get_all_convictions(self.db_path)
        name_width = max(len(name) for name in _DISPLAY_NAMES.values())
        lines = ["📊 Philosophical Conviction Scores", ""]
        for handle in _DISPLAY_ORDER:
            score = float(scores.get(handle, 0.5))
            filled = max(0, min(10, round(score * 10)))
            bar = "█" * filled + "░" * (10 - filled)
            if score < 0.35:
                label = "drifting toward doubt"
            elif score > 0.65:
                label = "doubling down"
            else:
                label = "holding position"
            display = _DISPLAY_NAMES[handle].ljust(name_width)
            lines.append(f"{display}  {bar}  {score:.2f}  {label}")
        await update.message.reply_text("\n".join(lines))

    async def _on_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None:
            return
        if self._wrong_chat(update):
            await update.message.reply_text(
                f"This chat's id is {update.message.chat_id}. "
                "Put that value in GROUP_CHAT_ID in .env and restart the process."
            )
            return
        await update.message.reply_text(
            "🤖 Philosophical Agent Chat\n"
            "\n"
            "Drop any topic and Nietzsche, Sartre, and Camus will debate it from their philosophical positions. Their conviction scores shift as the conversation evolves.\n"
            "\n"
            "/summary — Camus summarises the debate\n"
            "/convictions — Show each philosopher's current conviction score\n"
            "/reset — Clear conversation history and reset scores\n"
            "/help — This message"
        )

    async def start(self) -> None:
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        print("[Dispatcher] Listening for Telegram updates...")

    async def stop(self) -> None:
        await self.app.updater.stop()
        await self.app.stop()
        await self.app.shutdown()
