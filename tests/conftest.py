"""Unit tests import only pure modules (never polymarket_bot.py, which builds exchange clients
at import). The network is blocked and wallet/API credentials are removed from the environment."""
import os
import socket
import sys
from pathlib import Path

for k in ("PRIVATE_KEY", "WALLET_ADDRESS", "POLYMARKET_PROXY_ADDRESS", "MASTER_PASSWORD",
          "ANTHROPIC_API_KEY", "TG_BOT_TOKEN", "TG_CHAT_ID", "BOT_API_TOKEN"):
    os.environ.pop(k, None)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))


def _blocked(*args, **kwargs):
    raise RuntimeError("Network access is blocked in tests")


socket.socket.connect = _blocked
socket.create_connection = _blocked
