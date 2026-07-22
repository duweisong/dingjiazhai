"""
PushPlus connector — WeChat notification for trading signals.

PushPlus (http://www.pushplus.plus) is a free WeChat push service
widely used in Chinese quant communities. It sends messages to your
WeChat via a simple HTTP API.

Pattern extracted from etf_rotation_live.py.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)

PUSHPLUS_URL = "http://www.pushplus.plus/send"
CONFIG_FILE = Path(__file__).parent.parent.parent / "pushplus_config.json"


class PushPlusConnector:
    """Manages PushPlus token and message sending."""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or CONFIG_FILE
        self._config: Dict = {}
        self._load_config()

    def _load_config(self):
        """Load PushPlus configuration from JSON file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self._config = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load PushPlus config: {e}")
                self._config = {}

        # Defaults
        self._config.setdefault("pushplus_token", "")
        self._config.setdefault("notify_on_signal", True)
        self._config.setdefault("notify_on_error", True)
        self._config.setdefault("topic", "")

    @property
    def token(self) -> str:
        return self._config.get("pushplus_token", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.token)

    def send(
        self,
        title: str,
        content: str,
        template: str = "html",
        topic: Optional[str] = None,
    ) -> bool:
        """Send a PushPlus message.

        Args:
            title: Message title.
            content: Message body (HTML or plain text).
            template: "html" | "txt" | "json" | "markdown".
            topic: Optional topic ID for group push.

        Returns:
            True if sent successfully.
        """
        if not self.token:
            logger.info("[PushPlus] Token not configured — skipping push")
            return False

        payload = {
            "token": self.token,
            "title": title,
            "content": content,
            "template": template,
        }

        if topic:
            payload["topic"] = topic
        elif self._config.get("topic"):
            payload["topic"] = self._config["topic"]

        try:
            resp = requests.post(PUSHPLUS_URL, json=payload, timeout=15)
            result = resp.json()

            if result.get("code") == 200:
                logger.info(f"[PushPlus] Sent: {title}")
                return True
            else:
                logger.error(f"[PushPlus] Failed: {result.get('msg', 'Unknown error')}")
                return False
        except requests.RequestException as e:
            logger.error(f"[PushPlus] Network error: {e}")
            return False

    def send_signal_report(self, report) -> bool:
        """Send a weekly signal report via PushPlus.

        Args:
            report: WeeklySignalReport object with to_html() method.

        Returns:
            True if sent successfully.
        """
        title = f" 量化周报 {report.week_start.strftime('%m/%d')}-{report.week_end.strftime('%m/%d')}"
        content = report.to_html()
        return self.send(title, content, template="html")

    def send_alert(self, title: str, message: str) -> bool:
        """Send an urgent alert (risk limit breach, error, etc.)."""
        if not self._config.get("notify_on_error", True):
            return False
        content = f'<div style="color:red"><h3>⚠️ {title}</h3><p>{message}</p></div>'
        return self.send(f"⚠️ {title}", content, template="html")

    def setup_interactive(self):
        """Interactive setup for first-time PushPlus configuration."""
        if self.token:
            print(f"PushPlus already configured (token: {self.token[:8]}...)")
            return

        print("\n  PushPlus Setup")
        print("  " + "=" * 40)
        print("  1. Visit http://www.pushplus.plus and register")
        print("  2. Get your token from the dashboard")
        print("  3. Paste it below\n")

        token = input("  PushPlus Token (Enter to skip): ").strip()
        if token:
            self._config["pushplus_token"] = token
            self.save_config()
            print(f"  Saved to {self.config_path}")
        else:
            print("  Skipped. You can configure later in pushplus_config.json")

    def save_config(self):
        """Save current config to file."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)
        logger.info(f"PushPlus config saved to {self.config_path}")
