"""
Backtesting system for AgloK23 Trading Strategies.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class BacktestPosition:
    """Represents an open trading position."""
    symbol: str
    side: str  # 'buy' or 'sell'
    size: float
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    strategy_name: str = ""
    

@dataclass
class BacktestTrade:
    """Represents a completed trade."""
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    strategy_name: str = ""
    exit_reason: str = "manual"


@dataclass
class BacktestMetrics:
    """Performance metrics for a backtest."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)


class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission  # Commission as percentage of trade value
        
        # State variables
        self.current_capital = initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        self.trades: List[BacktestTrade] = []
        self.equity_history: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
    async def run_backtest(self, strategy, market_data: pd.DataFrame, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> BacktestMetrics:
        """
        Run backtest for a strategy using historical market data.
        
        Args:
            strategy: Strategy instance implementing BaseStrategy interface
            market_data: DataFrame with columns ['timestamp', 'symbol', 'price', 'volume']
            start_date: Start date for backtest (optional)
            end_date: End date for backtest (optional)
        """
        logger.info(f"Starting backtest for {strategy.name}")
        
        # Reset state
        self._reset_state()
        
        # Filter data by date range if provided
        data = market_data.copy()
        if 'timestamp' not in data.columns:
            raise ValueError("Market data must include 'timestamp' column")
        
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        if start_date:
            data = data[data['timestamp'] >= start_date]
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        if data.empty:
            raise ValueError("No data available for specified date range")
        
        # Start strategy
        await strategy.start()
        
        # Group data by timestamp for tick-by-tick processing
        data_grouped = data.groupby('timestamp')
        
        try:
            for timestamp, tick_data in data_grouped:
                # Prepare market data format for strategy
                market_snapshot = {}
                for _, row in tick_data.iterrows():
                    symbol = row['symbol']
                    market_snapshot[symbol] = {
                        'price': row['price'],
                        'volume': row['volume'] if 'volume' in row else 1000,
                        'timestamp': timestamp
                    }
                
                # Update positions with current prices
                await self._update_positions(market_snapshot, timestamp)
                
                # Generate signals from strategy
                signals = await strategy.generate_signals(market_snapshot)
                
                # Process signals
                if signals:
                    await self._process_signals(signals, market_snapshot, timestamp)
                
                # Record equity
                current_equity = self._calculate_current_equity(market_snapshot)
                self.equity_history.append((timestamp, current_equity))
                
                # Update peak and drawdown
                if current_equity > self.peak_equity:
                    self.peak_equity = current_equity
                
                drawdown = self.peak_equity - current_equity
                if drawdown > self.max_drawdown:
                    self.max_drawdown = drawdown
        
        finally:
            await strategy.stop()
        
        # Close any remaining positions at final prices
        if self.positions:
            final_prices = {symbol: data['price'] for symbol, data in market_snapshot.items()}
            final_timestamp = data['timestamp'].iloc[-1]
            await self._close_all_positions(final_prices, final_timestamp, "backtest_end")
        
        # Calculate final metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Backtest completed: {metrics.total_trades} trades, "
                   f"{metrics.win_rate:.1%} win rate, {metrics.total_return_pct:.2%} return")
        
        return metrics
    
    async def _update_positions(self, market_data: Dict[str, Dict], timestamp: datetime):
        """Update positions and check for stop loss/take profit triggers."""
        positions_to_close = []
        
        for symbol, position in self.positions.items():
            if symbol not in market_data:
                continue
                
            current_price = market_data[symbol]['price']
            
            # Check stop loss
            if position.stop_loss:
                if ((position.side == 'buy' and current_price <= position.stop_loss) or
                    (position.side == 'sell' and current_price >= position.stop_loss)):
                    positions_to_close.append((symbol, current_price, "stop_loss"))
                    continue
            
            # Check take profit
            if position.take_profit:
                if ((position.side == 'buy' and current_price >= position.take_profit) or
                    (position.side == 'sell' and current_price <= position.take_profit)):
                    positions_to_close.append((symbol, current_price, "take_profit"))
        
        # Close triggered positions
        for symbol, exit_price, reason in positions_to_close:
            await self._close_position(symbol, exit_price, timestamp, reason)
    
    async def _process_signals(self, signals: Dict[str, Dict], market_data: Dict[str, Dict], 
                             timestamp: datetime):
        """Process trading signals from strategy."""
        for symbol, signal in signals.items():
            if symbol not in market_data:
                logger.warning(f"Signal for {symbol} but no market data available")
                continue
            
            current_price = market_data[symbol]['price']
            side = signal.get('side')
            size = signal.get('size', 0.01)
            
            if side in ['buy', 'sell']:
                # Close existing position in same symbol first
                if symbol in self.positions:
                    await self._close_position(symbol, current_price, timestamp, "new_signal")
                
                # Open new position
                await self._open_position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    price=current_price,
                    timestamp=timestamp,
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    strategy_name=signal.get('strategy', 'unknown')
                )
    
    async def _open_position(self, symbol: str, side: str, size: float, price: float,
                           timestamp: datetime, stop_loss: Optional[float] = None,
                           take_profit: Optional[float] = None, strategy_name: str = ""):
        """Open a new trading position."""
        # Calculate position value
        position_value = abs(size) * self.current_capital
        commission_cost = position_value * self.commission
        
        # Check if we have enough capital
        if position_value + commission_cost > self.current_capital:
            logger.warning(f"Insufficient capital for {symbol} position: "
                          f"need ${position_value + commission_cost:.2f}, have ${self.current_capital:.2f}")
            return
        
        # Create position
        position = BacktestPosition(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=price,
            entry_time=timestamp,
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name
        )
        
        self.positions[symbol] = position
        
        # Deduct commission
        self.current_capital -= commission_cost
        
        logger.debug(f"Opened {side} position: {symbol} @ {price}, size: {size:.4f}")
    
    async def _close_position(self, symbol: str, exit_price: float, timestamp: datetime,
                            exit_reason: str = "manual"):
        """Close an existing position."""
        if symbol not in self.positions:
            return
        
        position = self.positions.pop(symbol)
        
        # Calculate P&L
        if position.side == 'buy':
            pnl_per_unit = exit_price - position.entry_price
        else:  # sell
            pnl_per_unit = position.entry_price - exit_price
        
        position_value = abs(position.size) * self.current_capital
        pnl = pnl_per_unit * position_value / position.entry_price
        pnl_pct = pnl_per_unit / position.entry_price
        
        # Deduct exit commission
        commission_cost = position_value * self.commission
        pnl -= commission_cost
        
        # Update capital
        self.current_capital += pnl
        
        # Create trade record
        trade = BacktestTrade(
            symbol=symbol,
            side=position.side,
            size=position.size,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_time=position.entry_time,
            exit_time=timestamp,
            pnl=pnl,
            pnl_pct=pnl_pct,
            strategy_name=position.strategy_name,
            exit_reason=exit_reason
        )
        
        self.trades.append(trade)
        
        logger.debug(f"Closed {position.side} position: {symbol} @ {exit_price}, "
                    f"P&L: ${pnl:.2f} ({pnl_pct:.2%})")
    
    async def _close_all_positions(self, final_prices: Dict[str, float], 
                                 timestamp: datetime, reason: str = "forced"):
        """Close all open positions."""
        symbols_to_close = list(self.positions.keys())
        
        for symbol in symbols_to_close:
            if symbol in final_prices:
                await self._close_position(symbol, final_prices[symbol], timestamp, reason)
    
    def _calculate_current_equity(self, market_data: Dict[str, Dict]) -> float:
        """Calculate current total equity including unrealized P&L."""
        equity = self.current_capital
        
        # Add unrealized P&L from open positions
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]['price']
                position_value = abs(position.size) * self.current_capital
                
                if position.side == 'buy':
                    unrealized_pnl = (current_price - position.entry_price) * position_value / position.entry_price
                else:
                    unrealized_pnl = (position.entry_price - current_price) * position_value / position.entry_price
                
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest performance metrics."""
        if not self.trades:
            return BacktestMetrics()
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L calculations
        total_pnl = sum(t.pnl for t in self.trades)
        total_return_pct = total_pnl / self.initial_capital
        
        # Average trade metrics
        avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        winning_pnls = [t.pnl for t in self.trades if t.pnl > 0]
        losing_pnls = [t.pnl for t in self.trades if t.pnl < 0]
        
        avg_winning_trade = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_losing_trade = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        
        # Risk metrics
        max_drawdown_pct = self.max_drawdown / self.initial_capital if self.initial_capital > 0 else 0
        
        # Profit factor
        gross_profit = sum(winning_pnls) if winning_pnls else 0
        gross_loss = abs(sum(losing_pnls)) if losing_pnls else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        # Sharpe and Sortino ratios
        if len(self.equity_history) > 1:
            returns = []
            for i in range(1, len(self.equity_history)):
                prev_equity = self.equity_history[i-1][1]
                curr_equity = self.equity_history[i][1]
                if prev_equity > 0:
                    returns.append((curr_equity - prev_equity) / prev_equity)
            
            if returns:
                returns_array = np.array(returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array)
                
                sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                
                # Sortino ratio (downside deviation)
                downside_returns = returns_array[returns_array < 0]
                downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
                sortino_ratio = mean_return / downside_std if downside_std > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        return BacktestMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return_pct=total_return_pct,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            avg_winning_trade=avg_winning_trade,
            avg_losing_trade=avg_losing_trade,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=self.trades.copy(),
            equity_curve=[eq[1] for eq in self.equity_history],
            timestamps=[eq[0] for eq in self.equity_history]
        )
    
    def _reset_state(self):
        """Reset backtester state for new backtest."""
        self.current_capital = self.initial_capital
        self.positions.clear()
        self.trades.clear()
        self.equity_history.clear()
        self.peak_equity = self.initial_capital
        self.max_drawdown = 0.0
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """Generate a formatted backtest report."""
        report = f"""
BACKTEST REPORT
===============

SUMMARY STATISTICS
------------------
Total Trades: {metrics.total_trades}
Winning Trades: {metrics.winning_trades}
Losing Trades: {metrics.losing_trades}
Win Rate: {metrics.win_rate:.2%}

RETURNS
-------
Total P&L: ${metrics.total_pnl:.2f}
Total Return: {metrics.total_return_pct:.2%}
Average Trade P&L: ${metrics.avg_trade_pnl:.2f}
Average Winning Trade: ${metrics.avg_winning_trade:.2f}
Average Losing Trade: ${metrics.avg_losing_trade:.2f}

RISK METRICS
------------
Max Drawdown: ${metrics.max_drawdown:.2f} ({metrics.max_drawdown_pct:.2%})
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Profit Factor: {metrics.profit_factor:.2f}

STREAKS
-------
Max Consecutive Wins: {metrics.max_consecutive_wins}
Max Consecutive Losses: {metrics.max_consecutive_losses}

TRADE DETAILS
-------------"""
        
        if metrics.trades:
            for i, trade in enumerate(metrics.trades[-10:], 1):  # Show last 10 trades
                report += f"""
Trade {len(metrics.trades) - 10 + i}: {trade.symbol} {trade.side.upper()}
  Entry: ${trade.entry_price:.2f} @ {trade.entry_time.strftime('%Y-%m-%d %H:%M')}
  Exit:  ${trade.exit_price:.2f} @ {trade.exit_time.strftime('%Y-%m-%d %H:%M')}
  P&L:   ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) [{trade.exit_reason}]"""
        
        return report
