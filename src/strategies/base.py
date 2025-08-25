"""
Base strategy interface for AgloK23 strategies.
"""
from __future__ import annotations

import abc
from typing import Dict, Any, Optional
from datetime import datetime

class BaseStrategy(abc.ABC):
    """Abstract base class for strategies."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        self.name = name
        self.params = params or {}
        self.running = False
        self.last_run: Optional[datetime] = None

    async def start(self):
        self.running = True

    async def stop(self):
        self.running = False

    @abc.abstractmethod
    async def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return signals dict, e.g., {symbol: {'side': 'buy', 'size': 1.0}}"""
        raise NotImplementedError

