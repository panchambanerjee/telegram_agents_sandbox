# Entry point: initialises DB, RAG, dispatcher, and agent loops

import warnings

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")
warnings.filterwarnings("ignore", message="fontTools is required")

import asyncio
import os
import signal
from dotenv import load_dotenv
load_dotenv()
from core.database import init_db
from core.dispatcher import Dispatcher
from rag.embedder import load_all_texts
from agents.personas import AGENTS


async def main() -> None:
    db_path = os.getenv("DB_PATH", "./data/conversations.db")
    await init_db(db_path)

    try:
        counts = load_all_texts(
            texts_path=os.getenv("TEXTS_PATH", "./texts"),
            chroma_path=os.getenv("CHROMA_PATH", "./data/chroma"),
        )
        for agent, n in counts.items():
            print(f"[RAG] {agent}: {n} chunks loaded")
    except Exception as e:
        print(f"[RAG] Warning: could not load texts — {e}")
        print("[RAG] Continuing without grounded passages.")

    agents = {}
    for handle, AgentClass in AGENTS.items():
        try:
            agents[handle] = AgentClass()
            print(f"[Main] {AgentClass.name} ready.")
        except ValueError as e:
            print(f"[Main] Skipping {handle}: {e}")

    if not agents:
        print("[Main] No agents loaded. Check your .env tokens.")
        return

    dispatcher = Dispatcher(agents)
    await dispatcher.start()

    tasks = [
        asyncio.create_task(agent.run(), name=agent.name)
        for agent in agents.values()
    ]
    print(f"\n✅ {len(agents)} philosopher(s) running. Drop a topic in your Telegram group.\n")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def _shutdown(sig):
        print(f"\n[Main] {sig.name} received, shutting down...")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _shutdown, sig)

    await stop_event.wait()

    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await dispatcher.stop()
    print("[Main] Shutdown complete.")


if __name__ == "__main__":
    asyncio.run(main())
