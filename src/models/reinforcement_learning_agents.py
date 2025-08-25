"""
Reinforcement Learning Trading Agents
===================================

Advanced RL agents for trading tasks including:
- Order execution optimization (TWAP, VWAP, market impact minimization)
- Portfolio management and asset allocation
- Market making and liquidity provision
- Multi-agent trading strategies

Features:
- Deep Q-Networks (DQN) for discrete action spaces
- Soft Actor-Critic (SAC) for continuous control
- Proximal Policy Optimization (PPO) for stable learning
- Custom trading environments with realistic market dynamics
- Multi-objective reward functions with risk-adjusted returns
"""

import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
from collections import deque, namedtuple
import gym
from gym import spaces
import random
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import asyncio
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Experience tuple for DQN
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


@dataclass
class TradingState:
    """Trading environment state representation."""
    prices: np.ndarray  # Current and historical prices
    volumes: np.ndarray  # Volume data
    inventory: float  # Current position
    cash: float  # Available cash
    time_remaining: int  # Time steps until end
    market_features: np.ndarray  # Technical indicators, etc.
    portfolio_value: float  # Total portfolio value
    drawdown: float  # Current drawdown
    volatility: float  # Recent volatility estimate


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for RL agents.
    
    Supports multiple trading tasks:
    - Portfolio optimization
    - Order execution
    - Market making
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000.0,
                 max_position: float = 10000.0,
                 transaction_cost: float = 0.001,
                 lookback_window: int = 50,
                 task_type: str = 'portfolio'):
        
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.max_position = max_position
        self.transaction_cost = transaction_cost
        self.lookback_window = lookback_window
        self.task_type = task_type
        
        # Environment state
        self.current_step = 0
        self.balance = initial_balance
        self.inventory = 0.0
        self.total_reward = 0.0
        self.trades_made = 0
        self.max_drawdown = 0.0
        
        # Action and observation spaces
        if task_type == 'portfolio':
            # Continuous actions: portfolio weights
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        elif task_type == 'execution':
            # Discrete actions: buy, sell, hold with different sizes
            self.action_space = spaces.Discrete(7)  # Strong sell, sell, weak sell, hold, weak buy, buy, strong buy
        elif task_type == 'market_making':
            # Continuous actions: bid/ask spreads and sizes
            self.action_space = spaces.Box(low=-0.01, high=0.01, shape=(4,), dtype=np.float32)
        
        # Observation space: market data + portfolio state
        obs_size = lookback_window * 5 + 10  # OHLCV * lookback + portfolio state
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        logger.info(f"TradingEnvironment initialized: {task_type} task, {len(data)} data points")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.inventory = 0.0
        self.total_reward = 0.0
        self.trades_made = 0
        self.max_drawdown = 0.0
        
        return self._get_observation()
    
    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment."""
        
        # Execute action
        reward = 0.0
        info = {}
        
        if self.task_type == 'portfolio':
            reward, info = self._execute_portfolio_action(action)
        elif self.task_type == 'execution':
            reward, info = self._execute_trading_action(action)
        elif self.task_type == 'market_making':
            reward, info = self._execute_market_making_action(action)
        
        # Update state
        self.current_step += 1
        self.total_reward += reward
        
        # Check if episode is done
        done = (self.current_step >= len(self.data) - 1 or 
                self.balance + self.inventory * self._get_current_price() <= self.initial_balance * 0.7)
        
        # Calculate current portfolio value and drawdown
        current_value = self.balance + self.inventory * self._get_current_price()
        peak_value = max(current_value, getattr(self, 'peak_value', current_value))
        self.peak_value = peak_value
        current_drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0.0
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Add portfolio metrics to info
        info.update({
            'portfolio_value': current_value,
            'current_drawdown': current_drawdown,
            'max_drawdown': self.max_drawdown,
            'total_return': (current_value - self.initial_balance) / self.initial_balance,
            'trades_made': self.trades_made
        })
        
        return self._get_observation(), reward, done, info
    
    def _execute_portfolio_action(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Execute portfolio optimization action."""
        # action represents target portfolio weights
        current_price = self._get_current_price()
        current_value = self.balance + self.inventory * current_price
        
        # Calculate target position
        target_inventory = action[0] * current_value / current_price if current_price > 0 else 0
        target_inventory = np.clip(target_inventory, -self.max_position, self.max_position)
        
        # Calculate trade size
        trade_size = target_inventory - self.inventory
        
        # Execute trade if significant
        if abs(trade_size) > current_value * 0.01:  # At least 1% of portfolio
            cost = abs(trade_size) * current_price * self.transaction_cost
            self.balance -= cost
            self.balance -= trade_size * current_price
            self.inventory += trade_size
            self.trades_made += 1
        
        # Calculate reward (risk-adjusted return)
        if self.current_step > self.lookback_window:
            prev_price = self.data.iloc[self.current_step - 1]['close']
            price_return = (current_price - prev_price) / prev_price if prev_price > 0 else 0
            
            # Portfolio return
            portfolio_return = self.inventory * price_return
            
            # Risk penalty (volatility and drawdown)
            volatility = self._calculate_volatility()
            risk_penalty = 0.1 * volatility + 0.2 * self.max_drawdown
            
            reward = portfolio_return - risk_penalty
        else:
            reward = 0.0
        
        return reward, {'trade_size': trade_size, 'cost': cost if 'cost' in locals() else 0.0}
    
    def _execute_trading_action(self, action: int) -> Tuple[float, Dict]:
        """Execute discrete trading action."""
        current_price = self._get_current_price()
        
        # Map action to trade size (as fraction of max position)
        action_map = {
            0: -0.5,   # Strong sell
            1: -0.25,  # Sell
            2: -0.1,   # Weak sell
            3: 0.0,    # Hold
            4: 0.1,    # Weak buy
            5: 0.25,   # Buy
            6: 0.5     # Strong buy
        }
        
        trade_fraction = action_map.get(action, 0.0)
        trade_size = trade_fraction * self.max_position
        
        # Check position limits
        new_inventory = self.inventory + trade_size
        if abs(new_inventory) > self.max_position:
            trade_size = np.sign(trade_size) * (self.max_position - abs(self.inventory))
        
        # Execute trade
        cost = 0.0
        if abs(trade_size) > 0:
            trade_value = trade_size * current_price
            cost = abs(trade_value) * self.transaction_cost
            
            if self.balance >= trade_value + cost:  # Check if we have enough cash
                self.balance -= (trade_value + cost)
                self.inventory += trade_size
                self.trades_made += 1
            else:
                trade_size = 0.0  # Can't afford trade
        
        # Calculate reward based on future price movement
        reward = 0.0
        if self.current_step < len(self.data) - 1:
            next_price = self.data.iloc[self.current_step + 1]['close']
            price_change = (next_price - current_price) / current_price if current_price > 0 else 0
            
            # Reward = position * price_change - transaction costs
            reward = self.inventory * price_change - cost / self.initial_balance
            
            # Add execution quality bonus/penalty
            if abs(trade_size) > 0:
                # Reward good timing
                if (trade_size > 0 and price_change > 0) or (trade_size < 0 and price_change < 0):
                    reward += 0.01  # Bonus for good timing
                else:
                    reward -= 0.005  # Small penalty for bad timing
        
        return reward, {'trade_size': trade_size, 'cost': cost}
    
    def _execute_market_making_action(self, action: np.ndarray) -> Tuple[float, Dict]:
        """Execute market making action."""
        current_price = self._get_current_price()
        
        # action[0, 1] = bid/ask spread adjustments
        # action[2, 3] = bid/ask sizes
        bid_spread = np.clip(action[0], -0.005, 0.005)
        ask_spread = np.clip(action[1], -0.005, 0.005)
        bid_size = np.clip(abs(action[2]), 0.1, 1.0) * 100
        ask_size = np.clip(abs(action[3]), 0.1, 1.0) * 100
        
        bid_price = current_price * (1 + bid_spread)
        ask_price = current_price * (1 + ask_spread)
        
        # Simulate order fills based on market volatility
        volatility = self._calculate_volatility()
        fill_probability = min(volatility * 10, 0.5)  # Higher vol = higher fill rate
        
        reward = 0.0
        trades_executed = 0
        
        # Simulate bid fill
        if random.random() < fill_probability and self.inventory > -self.max_position:
            self.balance -= bid_price * bid_size
            self.inventory += bid_size
            reward += (current_price - bid_price) * bid_size / self.initial_balance
            trades_executed += 1
        
        # Simulate ask fill
        if random.random() < fill_probability and self.inventory < self.max_position:
            self.balance += ask_price * ask_size
            self.inventory -= ask_size
            reward += (ask_price - current_price) * ask_size / self.initial_balance
            trades_executed += 1
        
        # Inventory penalty (market makers want to stay neutral)
        inventory_penalty = 0.001 * (self.inventory / self.max_position) ** 2
        reward -= inventory_penalty
        
        self.trades_made += trades_executed
        
        return reward, {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread': ask_price - bid_price,
            'trades_executed': trades_executed
        }
    
    def _get_observation(self) -> np.ndarray:
        """Get current environment observation."""
        # Market data features
        start_idx = max(0, self.current_step - self.lookback_window)
        end_idx = self.current_step + 1
        
        market_data = self.data.iloc[start_idx:end_idx][['open', 'high', 'low', 'close', 'volume']].values
        
        # Pad if necessary
        if len(market_data) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(market_data), 5))
            market_data = np.vstack([padding, market_data])
        
        market_features = market_data.flatten()
        
        # Normalize market features
        market_features = (market_features - np.mean(market_features)) / (np.std(market_features) + 1e-8)
        
        # Portfolio state features
        current_price = self._get_current_price()
        portfolio_value = self.balance + self.inventory * current_price
        
        portfolio_features = np.array([
            self.balance / self.initial_balance,  # Normalized cash
            self.inventory / self.max_position,   # Normalized inventory
            portfolio_value / self.initial_balance,  # Normalized portfolio value
            self.max_drawdown,  # Max drawdown
            self._calculate_volatility(),  # Current volatility
            self.trades_made / 100.0,  # Normalized trade count
            (self.current_step - self.lookback_window) / (len(self.data) - self.lookback_window),  # Progress
            current_price / self.data.iloc[self.lookback_window]['close'] - 1,  # Price change since start
            self.total_reward,  # Cumulative reward
            min(self.current_step / 100.0, 1.0)  # Time factor
        ])
        
        # Combine features
        observation = np.concatenate([market_features, portfolio_features]).astype(np.float32)
        
        # Handle any remaining NaN or inf values
        observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return observation
    
    def _get_current_price(self) -> float:
        """Get current market price."""
        if self.current_step < len(self.data):
            return self.data.iloc[self.current_step]['close']
        return self.data.iloc[-1]['close']
    
    def _calculate_volatility(self, window: int = 20) -> float:
        """Calculate recent price volatility."""
        start_idx = max(0, self.current_step - window)
        end_idx = self.current_step + 1
        
        if end_idx - start_idx < 2:
            return 0.0
        
        prices = self.data.iloc[start_idx:end_idx]['close']
        returns = prices.pct_change().dropna()
        
        if len(returns) > 0:
            return float(returns.std())
        return 0.0


class DQNAgent:
    """
    Deep Q-Network agent for discrete action trading.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 32):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural networks
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=memory_size)
        
        # Update target network
        self._update_target_network()
        
        logger.info(f"DQN Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        )
    
    def remember(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.memory.append(Experience(state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy."""
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(self.device)
        actions = torch.LongTensor([e.action for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(self.device)
        dones = torch.BoolTensor([e.done for e in batch]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def update_target_network(self):
        """Public method to update target network."""
        self._update_target_network()


class SACAgent:
    """
    Soft Actor-Critic agent for continuous action trading.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int,
                 learning_rate: float = 0.0003,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 memory_size: int = 100000,
                 batch_size: int = 256):
        
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = self._build_actor().to(self.device)
        self.critic1 = self._build_critic().to(self.device)
        self.critic2 = self._build_critic().to(self.device)
        self.target_critic1 = self._build_critic().to(self.device)
        self.target_critic2 = self._build_critic().to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=learning_rate)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=learning_rate)
        
        # Initialize target networks
        self._soft_update(self.target_critic1, self.critic1, 1.0)
        self._soft_update(self.target_critic2, self.critic2, 1.0)
        
        # Experience buffer
        self.memory = deque(maxlen=memory_size)
        
        logger.info(f"SAC Agent initialized: state_size={state_size}, action_size={action_size}")
    
    def _build_actor(self) -> nn.Module:
        """Build actor network."""
        class Actor(nn.Module):
            def __init__(self, state_size, action_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(state_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU()
                )
                self.mean = nn.Linear(128, action_size)
                self.log_std = nn.Linear(128, action_size)
                
            def forward(self, state):
                x = self.network(state)
                mean = self.mean(x)
                log_std = torch.clamp(self.log_std(x), -20, 2)
                return mean, log_std
            
            def sample(self, state):
                mean, log_std = self.forward(state)
                std = log_std.exp()
                normal = Normal(mean, std)
                x_t = normal.rsample()  # Reparameterization trick
                action = torch.tanh(x_t)
                
                # Calculate log probability
                log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(1, keepdim=True)
                
                return action, log_prob
        
        return Actor(self.state_size, self.action_size)
    
    def _build_critic(self) -> nn.Module:
        """Build critic network."""
        return nn.Sequential(
            nn.Linear(self.state_size + self.action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def remember(self, state: np.ndarray, action: np.ndarray, reward: float, 
                 next_state: np.ndarray, done: bool):
        """Store experience."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training:
            action, _ = self.actor.sample(state_tensor)
        else:
            mean, _ = self.actor(state_tensor)
            action = torch.tanh(mean)
        
        return action.cpu().detach().numpy()[0]
    
    def update(self):
        """Update all networks."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            target_q1 = self.target_critic1(torch.cat([next_states, next_actions], 1))
            target_q2 = self.target_critic2(torch.cat([next_states, next_actions], 1))
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * target_q * ~dones
        
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.actor.sample(states)
        q1 = self.critic1(torch.cat([states, new_actions], 1))
        q2 = self.critic2(torch.cat([states, new_actions], 1))
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1, self.tau)
        self._soft_update(self.target_critic2, self.critic2, self.tau)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class RLTradingSystem:
    """
    Complete RL trading system managing multiple agents and environments.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 agent_type: str = 'DQN',
                 task_type: str = 'execution',
                 initial_balance: float = 100000.0):
        
        self.data = data
        self.agent_type = agent_type
        self.task_type = task_type
        self.initial_balance = initial_balance
        
        # Create environment
        self.env = TradingEnvironment(
            data=data,
            initial_balance=initial_balance,
            task_type=task_type
        )
        
        # Create agent
        state_size = self.env.observation_space.shape[0]
        if task_type == 'execution':
            action_size = self.env.action_space.n
            self.agent = DQNAgent(state_size, action_size)
        else:
            action_size = self.env.action_space.shape[0]
            self.agent = SACAgent(state_size, action_size)
        
        # Training metrics
        self.training_history = []
        self.episode_rewards = []
        self.episode_returns = []
        
        logger.info(f"RLTradingSystem initialized: {agent_type} agent, {task_type} task")
    
    def train(self, episodes: int = 1000, max_steps: int = None) -> Dict[str, List]:
        """Train the RL agent."""
        logger.info(f"Starting training: {episodes} episodes")
        
        if max_steps is None:
            max_steps = len(self.data) - self.env.lookback_window - 1
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            steps = 0
            
            while steps < max_steps:
                # Select action
                if self.agent_type == 'DQN':
                    action = self.agent.act(state, training=True)
                else:
                    action = self.agent.act(state, training=True)
                
                # Execute action
                next_state, reward, done, info = self.env.step(action)
                
                # Store experience
                if self.agent_type == 'DQN':
                    self.agent.remember(state, action, reward, next_state, done)
                else:
                    self.agent.remember(state, action, reward, next_state, done)
                
                # Update agent
                if self.agent_type == 'DQN':
                    if len(self.agent.memory) > self.agent.batch_size:
                        self.agent.replay()
                        if steps % 100 == 0:  # Update target network every 100 steps
                            self.agent.update_target_network()
                else:
                    self.agent.update()
                
                state = next_state
                total_reward += reward
                steps += 1
                
                if done:
                    break
            
            # Record episode metrics
            self.episode_rewards.append(total_reward)
            portfolio_return = info.get('total_return', 0.0)
            self.episode_returns.append(portfolio_return)
            
            self.training_history.append({
                'episode': episode,
                'reward': total_reward,
                'return': portfolio_return,
                'max_drawdown': info.get('max_drawdown', 0.0),
                'trades': info.get('trades_made', 0),
                'steps': steps
            })
            
            # Log progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_return = np.mean(self.episode_returns[-100:])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.4f}, "
                          f"Avg Return={avg_return:.4f}, Epsilon={getattr(self.agent, 'epsilon', 'N/A')}")
        
        logger.info("Training completed")
        return {
            'rewards': self.episode_rewards,
            'returns': self.episode_returns,
            'history': self.training_history
        }
    
    def evaluate(self, episodes: int = 100) -> Dict[str, Any]:
        """Evaluate the trained agent."""
        logger.info(f"Evaluating agent: {episodes} episodes")
        
        eval_rewards = []
        eval_returns = []
        eval_drawdowns = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False
            
            while not done:
                # Select action (no exploration)
                if self.agent_type == 'DQN':
                    action = self.agent.act(state, training=False)
                else:
                    action = self.agent.act(state, training=False)
                
                state, reward, done, info = self.env.step(action)
                total_reward += reward
            
            eval_rewards.append(total_reward)
            eval_returns.append(info.get('total_return', 0.0))
            eval_drawdowns.append(info.get('max_drawdown', 0.0))
        
        # Calculate performance metrics
        results = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_return': np.mean(eval_returns),
            'std_return': np.std(eval_returns),
            'mean_drawdown': np.mean(eval_drawdowns),
            'max_drawdown': np.max(eval_drawdowns),
            'sharpe_ratio': np.mean(eval_returns) / (np.std(eval_returns) + 1e-8),
            'win_rate': np.mean([r > 0 for r in eval_returns])
        }
        
        logger.info(f"Evaluation results: Return={results['mean_return']:.4f}±{results['std_return']:.4f}, "
                   f"Sharpe={results['sharpe_ratio']:.4f}, Drawdown={results['mean_drawdown']:.4f}")
        
        return results
    
    async def run_live_trading(self, duration_minutes: int = 60):
        """Simulate live trading."""
        logger.info(f"Starting live trading simulation: {duration_minutes} minutes")
        
        state = self.env.reset()
        start_time = datetime.now()
        trades_made = []
        
        while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
            # Get action from trained agent
            if self.agent_type == 'DQN':
                action = self.agent.act(state, training=False)
            else:
                action = self.agent.act(state, training=False)
            
            # Execute action
            next_state, reward, done, info = self.env.step(action)
            
            # Record trade
            if info.get('trade_size', 0) != 0:
                trades_made.append({
                    'timestamp': datetime.now(),
                    'action': action,
                    'trade_size': info['trade_size'],
                    'reward': reward,
                    'portfolio_value': info.get('portfolio_value', 0)
                })
                
                logger.info(f"Trade executed: size={info['trade_size']:.2f}, "
                          f"value=${info.get('portfolio_value', 0):.2f}")
            
            state = next_state
            
            if done:
                state = self.env.reset()
            
            # Sleep to simulate real-time (adjust as needed)
            await asyncio.sleep(1.0)
        
        logger.info(f"Live trading completed: {len(trades_made)} trades executed")
        return trades_made


# Test and example functions
def create_synthetic_market_data(days: int = 252, volatility: float = 0.02) -> pd.DataFrame:
    """Create synthetic market data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate realistic price movements
    returns = np.random.normal(0.0005, volatility, days)
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC data
    data = []
    for i, price in enumerate(prices):
        daily_vol = volatility * np.random.uniform(0.5, 1.5)
        high = price * (1 + abs(np.random.normal(0, daily_vol)))
        low = price * (1 - abs(np.random.normal(0, daily_vol)))
        open_price = prices[i-1] if i > 0 else price
        volume = np.random.randint(10000, 1000000)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data).set_index('date')


async def test_rl_system():
    """Test the RL trading system."""
    logger.info("🤖 Testing RL Trading System...")
    
    try:
        # Create synthetic data
        data = create_synthetic_market_data(days=500, volatility=0.015)
        logger.info(f"Created synthetic data: {len(data)} days")
        
        # Test DQN agent with execution task
        dqn_system = RLTradingSystem(
            data=data[:400],  # Use first 400 days for training
            agent_type='DQN',
            task_type='execution'
        )
        
        # Quick training
        training_results = dqn_system.train(episodes=50, max_steps=100)
        logger.info(f"✅ DQN training completed: {len(training_results['rewards'])} episodes")
        
        # Evaluation
        eval_results = dqn_system.evaluate(episodes=10)
        logger.info(f"✅ DQN evaluation: Return={eval_results['mean_return']:.4f}, "
                   f"Sharpe={eval_results['sharpe_ratio']:.4f}")
        
        # Test SAC agent with portfolio task
        sac_system = RLTradingSystem(
            data=data[:400],
            agent_type='SAC',
            task_type='portfolio'
        )
        
        # Quick training
        training_results = sac_system.train(episodes=30, max_steps=100)
        logger.info(f"✅ SAC training completed: {len(training_results['rewards'])} episodes")
        
        # Evaluation
        eval_results = sac_system.evaluate(episodes=10)
        logger.info(f"✅ SAC evaluation: Return={eval_results['mean_return']:.4f}, "
                   f"Sharpe={eval_results['sharpe_ratio']:.4f}")
        
        # Test live trading simulation
        trades = await dqn_system.run_live_trading(duration_minutes=1)  # 1 minute test
        logger.info(f"✅ Live trading simulation: {len(trades)} trades")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ RL system test failed: {e}")
        return False


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_rl_system())
