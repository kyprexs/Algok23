"""
Strategy Engine for AgloK23 Trading System
==========================================

Coordinates multiple trading strategies and generates trading signals.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.config.settings import Settings
from .base import BaseStrategy
from . import get_strategy

logger = logging.getLogger(__name__)


class StrategyEngine:
    """Trading strategy coordination and signal generation."""
    
    def __init__(self, settings: Settings, model_manager, risk_manager, order_manager):
        self.settings = settings
        self.model_manager = model_manager
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.running = False
        self.active_strategies: List[BaseStrategy] = []
        self.strategy_configs = {}
        self.last_market_data = {}
        
    async def start(self):
        """Start the strategy engine."""
        logger.info("🎯 Starting Strategy Engine...")
        self.running = True
        
        # Initialize and start strategies
        await self._initialize_strategies()
        
        logger.info(f"✅ Strategy Engine started with {len(self.active_strategies)} strategies")
    
    async def stop(self):
        """Stop the strategy engine."""
        logger.info("🛑 Stopping Strategy Engine...")
        self.running = False
        
        # Stop all active strategies
        for strategy in self.active_strategies:
            try:
                await strategy.stop()
                logger.info(f"Stopped strategy: {strategy.name}")
            except Exception as e:
                logger.error(f"Error stopping strategy {strategy.name}: {e}")
        
        self.active_strategies.clear()
        logger.info("✅ Strategy Engine stopped")
    
    async def add_strategy(self, strategy_name: str, params: Optional[Dict[str, Any]] = None, 
                          name: Optional[str] = None) -> bool:
        """Add a strategy to the engine."""
        try:
            strategy = get_strategy(strategy_name, name=name or f"{strategy_name}_{len(self.active_strategies)}", 
                                 params=params)
            
            if self.running:
                await strategy.start()
            
            self.active_strategies.append(strategy)
            self.strategy_configs[strategy.name] = {
                'strategy_type': strategy_name,
                'params': params or {},
                'added_at': datetime.utcnow().isoformat()
            }
            
            logger.info(f"Added strategy: {strategy.name} ({strategy_name})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding strategy {strategy_name}: {e}")
            return False
    
    async def remove_strategy(self, strategy_name: str) -> bool:
        """Remove a strategy from the engine."""
        try:
            for i, strategy in enumerate(self.active_strategies):
                if strategy.name == strategy_name:
                    await strategy.stop()
                    self.active_strategies.pop(i)
                    self.strategy_configs.pop(strategy_name, None)
                    logger.info(f"Removed strategy: {strategy_name}")
                    return True
            
            logger.warning(f"Strategy not found: {strategy_name}")
            return False
            
        except Exception as e:
            logger.error(f"Error removing strategy {strategy_name}: {e}")
            return False
    
    async def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data for all strategies."""
        self.last_market_data = market_data
        
        if not self.running:
            return
        
        # Process market data through strategies
        await self.generate_signals(market_data)
    
    async def generate_signals(self, market_data: Optional[Dict[str, Any]] = None):
        """Generate trading signals from all active strategies."""
        if not self.running or not self.active_strategies:
            return {}
        
        try:
            data_to_use = market_data or self.last_market_data
            if not data_to_use:
                logger.debug("No market data available for signal generation")
                return {}
            
            all_signals = {}
            
            # Generate signals from all strategies concurrently
            tasks = []
            for strategy in self.active_strategies:
                task = asyncio.create_task(self._safe_generate_signals(strategy, data_to_use))
                tasks.append((strategy.name, task))
            
            # Collect results
            for strategy_name, task in tasks:
                try:
                    signals = await task
                    if signals:
                        all_signals[strategy_name] = signals
                        logger.debug(f"Generated {len(signals)} signals from {strategy_name}")
                except Exception as e:
                    logger.error(f"Error getting signals from {strategy_name}: {e}")
            
            # Process signals through risk management
            if all_signals:
                processed_signals = await self._process_signals_through_risk_management(all_signals)
                return processed_signals
                
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return {}
    
    async def _safe_generate_signals(self, strategy: BaseStrategy, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safely generate signals from a strategy."""
        try:
            return await strategy.generate_signals(market_data)
        except Exception as e:
            logger.error(f"Error generating signals from {strategy.name}: {e}")
            return {}
    
    async def _process_signals_through_risk_management(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Process signals through risk management filters."""
        try:
            # Flatten signals by symbol
            symbol_signals = {}
            
            for strategy_name, strategy_signals in signals.items():
                for symbol, signal in strategy_signals.items():
                    if symbol not in symbol_signals:
                        symbol_signals[symbol] = []
                    
                    signal['strategy'] = strategy_name
                    symbol_signals[symbol].append(signal)
            
            # Process through risk manager if available
            if self.risk_manager and hasattr(self.risk_manager, 'filter_signals'):
                filtered_signals = await self.risk_manager.filter_signals(symbol_signals)
                return filtered_signals
            
            return symbol_signals
            
        except Exception as e:
            logger.error(f"Error processing signals through risk management: {e}")
            return signals
    
    async def _initialize_strategies(self):
        """Initialize strategies from settings."""
        try:
            # Get strategy configurations from settings if available
            if hasattr(self.settings, 'strategies') and self.settings.strategies:
                for strategy_config in self.settings.strategies:
                    strategy_name = strategy_config.get('name', 'momentum')
                    params = strategy_config.get('params', {})
                    instance_name = strategy_config.get('instance_name')
                    
                    await self.add_strategy(strategy_name, params, instance_name)
            else:
                # Default strategies if none configured
                await self.add_strategy('momentum', name='default_momentum')
                await self.add_strategy('mean_reversion', name='default_mean_reversion')
                
        except Exception as e:
            logger.error(f"Error initializing strategies: {e}")
    
    async def get_strategy_status(self) -> Dict[str, Any]:
        """Get status of all active strategies."""
        strategy_status = []
        
        for strategy in self.active_strategies:
            status = {
                'name': strategy.name,
                'type': strategy.__class__.__name__,
                'running': strategy.running,
                'params': strategy.params,
                'last_run': strategy.last_run.isoformat() if strategy.last_run else None
            }
            
            # Add additional metrics if available
            if hasattr(strategy, 'get_additional_metrics'):
                try:
                    for symbol in self.last_market_data.keys():
                        metrics = strategy.get_additional_metrics(symbol)
                        if metrics:
                            status[f'{symbol}_metrics'] = metrics
                except:
                    pass  # Metrics are optional
            
            strategy_status.append(status)
        
        return {
            'running': self.running,
            'total_strategies': len(self.active_strategies),
            'strategies': strategy_status,
            'last_market_data_update': datetime.utcnow().isoformat() if self.last_market_data else None,
            'timestamp': datetime.utcnow().isoformat()
        }
