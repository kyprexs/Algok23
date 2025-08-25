"""
Configuration Management for AgloK23 Trading System
===================================================

Centralized configuration using Pydantic Settings with environment variable support.
Provides type-safe configuration with validation and defaults.
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
from enum import Enum


class TradingMode(str, Enum):
    """Trading mode enumeration."""
    PAPER = "paper"
    LIVE = "live"


class Environment(str, Enum):
    """Environment enumeration."""
    DEVELOPMENT = "development" 
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"


class Settings(BaseSettings):
    """
    Main configuration class for AgloK23 trading system.
    
    Uses Pydantic BaseSettings to load configuration from environment variables
    with proper type validation and sensible defaults.
    """
    
    # =============================================================================
    # TRADING ENVIRONMENT
    # =============================================================================
    TRADING_MODE: TradingMode = TradingMode.PAPER
    ENVIRONMENT: Environment = Environment.DEVELOPMENT
    LOG_LEVEL: LogLevel = LogLevel.INFO
    DEBUG_MODE: bool = True
    
    # =============================================================================
    # CRYPTOCURRENCY EXCHANGE APIs
    # =============================================================================
    
    # Binance Configuration
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_SECRET_KEY: Optional[str] = None
    BINANCE_SANDBOX: bool = True
    BINANCE_BASE_URL: str = "https://testnet.binance.vision/api/v3"
    
    # Coinbase Configuration  
    COINBASE_API_KEY: Optional[str] = None
    COINBASE_SECRET_KEY: Optional[str] = None
    COINBASE_PASSPHRASE: Optional[str] = None
    COINBASE_SANDBOX: bool = True
    COINBASE_BASE_URL: str = "https://api-public.sandbox.exchange.coinbase.com"
    
    # =============================================================================
    # EQUITY MARKET APIs
    # =============================================================================
    
    # Polygon.io Configuration
    POLYGON_API_KEY: Optional[str] = None
    POLYGON_BASE_URL: str = "https://api.polygon.io"
    
    # Interactive Brokers Configuration
    IB_HOST: str = "127.0.0.1"
    IB_PORT: int = 7497  # Paper trading port
    IB_CLIENT_ID: int = 1
    IB_TIMEOUT: int = 60
    
    # Alpaca Configuration
    ALPACA_API_KEY: Optional[str] = None
    ALPACA_SECRET_KEY: Optional[str] = None
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # =============================================================================
    # ALTERNATIVE DATA SOURCES
    # =============================================================================
    GLASSNODE_API_KEY: Optional[str] = None
    NEWS_API_KEY: Optional[str] = None
    TWITTER_BEARER_TOKEN: Optional[str] = None
    TWITTER_API_KEY: Optional[str] = None
    TWITTER_API_SECRET: Optional[str] = None
    TWITTER_ACCESS_TOKEN: Optional[str] = None
    TWITTER_ACCESS_SECRET: Optional[str] = None
    REDDIT_CLIENT_ID: Optional[str] = None
    REDDIT_CLIENT_SECRET: Optional[str] = None
    REDDIT_USER_AGENT: str = "AgloK23TradingBot/1.0"
    ALPHA_VANTAGE_API_KEY: Optional[str] = None
    QUANDL_API_KEY: Optional[str] = None
    
    # =============================================================================
    # DATABASE CONFIGURATION
    # =============================================================================
    DATABASE_URL: str = "postgresql://postgres:password@localhost:5432/algok23"
    TIMESCALEDB_URL: str = "postgresql://postgres:password@localhost:5432/algok23_timeseries"
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    
    # =============================================================================
    # MESSAGE QUEUE & STREAMING
    # =============================================================================
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_CONSUMER_GROUP: str = "algok23-consumers"
    KAFKA_ACKS: str = "all"
    KAFKA_RETRIES: int = 3
    
    # =============================================================================
    # MACHINE LEARNING & MODEL REGISTRY
    # =============================================================================
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "algok23-trading-models"
    MODEL_REGISTRY_PATH: str = "./models"
    FEATURE_STORE_PATH: str = "./features"
    
    # =============================================================================
    # RISK MANAGEMENT SETTINGS
    # =============================================================================
    MAX_PORTFOLIO_DRAWDOWN: float = Field(0.20, ge=0.01, le=0.50)
    MAX_ASSET_DRAWDOWN: float = Field(0.10, ge=0.01, le=0.30)
    DAILY_LOSS_LIMIT: float = Field(0.05, ge=0.01, le=0.20)
    MAX_POSITION_SIZE: float = Field(0.05, ge=0.001, le=0.20)
    MAX_SECTOR_EXPOSURE: float = Field(0.20, ge=0.05, le=0.50)
    MAX_CRYPTO_LEVERAGE: float = Field(3.0, ge=1.0, le=10.0)
    MAX_EQUITY_LEVERAGE: float = Field(2.0, ge=1.0, le=5.0)
    POSITION_SIZING_METHOD: str = "volatility_adjusted"
    ATR_MULTIPLIER: float = Field(2.0, ge=0.5, le=5.0)
    KELLY_FRACTION_CAP: float = Field(0.25, ge=0.01, le=0.50)
    
    # =============================================================================
    # TRADING STRATEGY PARAMETERS
    # =============================================================================
    ACTIVE_STRATEGIES: str = "momentum_breakout,mean_reversion,pairs_trading"
    REGIME_DETECTION_ENABLED: bool = True
    MULTI_TIMEFRAME_ENABLED: bool = True
    PRIMARY_TIMEFRAME: str = "5m"
    SIGNAL_TIMEFRAMES: str = "1m,5m,15m,1h,1d"
    LOOKBACK_PERIODS: int = Field(252, ge=50, le=1000)
    
    # Technical Indicators
    RSI_PERIOD: int = Field(14, ge=5, le=50)
    MACD_FAST: int = Field(12, ge=5, le=25)
    MACD_SLOW: int = Field(26, ge=15, le=50)
    MACD_SIGNAL: int = Field(9, ge=5, le=20)
    BB_PERIOD: int = Field(20, ge=10, le=50)
    BB_STD: float = Field(2.0, ge=1.0, le=3.0)
    ATR_PERIOD: int = Field(14, ge=5, le=30)
    
    # =============================================================================
    # EXECUTION SETTINGS
    # =============================================================================
    DEFAULT_ORDER_TYPE: str = "limit"
    SLIPPAGE_TOLERANCE: float = Field(0.001, ge=0.0001, le=0.01)
    MAX_ORDER_SIZE_USD: float = Field(10000.0, ge=100.0, le=1000000.0)
    MIN_ORDER_SIZE_USD: float = Field(10.0, ge=1.0, le=100.0)
    USE_SMART_ROUTING: bool = True
    ICEBERG_ORDER_ENABLED: bool = True
    VWAP_SLICING_ENABLED: bool = True
    MAX_ORDER_CHUNKS: int = Field(5, ge=1, le=20)
    MAX_LATENCY_MS: int = Field(100, ge=10, le=1000)
    HEARTBEAT_INTERVAL: int = Field(30, ge=5, le=300)
    
    # =============================================================================
    # MONITORING & ALERTING
    # =============================================================================
    GRAFANA_URL: str = "http://localhost:3000"
    GRAFANA_USERNAME: str = "admin"
    GRAFANA_PASSWORD: str = "admin"
    PROMETHEUS_URL: str = "http://localhost:9090"
    METRICS_PORT: int = 8000
    SLACK_WEBHOOK_URL: Optional[str] = None
    EMAIL_SMTP_SERVER: str = "smtp.gmail.com"
    EMAIL_SMTP_PORT: int = 587
    EMAIL_USERNAME: Optional[str] = None
    EMAIL_PASSWORD: Optional[str] = None
    ALERT_EMAIL_TO: Optional[str] = None
    SENTRY_DSN: Optional[str] = None
    
    # =============================================================================
    # CLOUD & INFRASTRUCTURE
    # =============================================================================
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    AWS_S3_BUCKET: Optional[str] = None
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    K8S_NAMESPACE: str = "algok23"
    K8S_CONFIG_PATH: str = "~/.kube/config"
    
    # =============================================================================
    # DEVELOPMENT & DEBUGGING
    # =============================================================================
    JUPYTER_PORT: int = 8888
    JUPYTER_TOKEN: Optional[str] = None
    DEV_DATABASE_URL: str = "sqlite:///./algok23_dev.db"
    TEST_DATABASE_URL: str = "sqlite:///./algok23_test.db"
    PYTEST_PARALLEL: int = 4
    ENABLE_PROFILING: bool = False
    PROFILING_OUTPUT_DIR: str = "./profiling"
    
    # =============================================================================
    # BACKUP & DISASTER RECOVERY
    # =============================================================================
    BACKUP_INTERVAL_HOURS: int = Field(6, ge=1, le=24)
    BACKUP_RETENTION_DAYS: int = Field(30, ge=1, le=365)
    BACKUP_S3_BUCKET: Optional[str] = None
    MODEL_BACKUP_ENABLED: bool = True
    MODEL_VERSIONING_ENABLED: bool = True
    
    # =============================================================================
    # COMPLIANCE & AUDIT
    # =============================================================================
    TRADE_LOG_RETENTION_DAYS: int = Field(2555, ge=365, le=3650)  # 7 years
    AUDIT_LOG_ENABLED: bool = True
    TRADE_REPORTING_ENABLED: bool = True
    DATA_RETENTION_DAYS: int = Field(1095, ge=365, le=2555)  # 3 years
    LOG_RETENTION_DAYS: int = Field(365, ge=30, le=1095)  # 1 year
    ENCRYPT_SENSITIVE_DATA: bool = True
    AUDIT_API_CALLS: bool = True
    RATE_LIMITING_ENABLED: bool = True
    
    # =============================================================================
    # FEATURE FLAGS
    # =============================================================================
    ENABLE_RL_TRADING: bool = False
    ENABLE_NLP_SENTIMENT: bool = True
    ENABLE_CROSS_ASSET_SIGNALS: bool = True
    ENABLE_OPTIONS_TRADING: bool = False
    ENABLE_FUTURES_TRADING: bool = True
    ENABLE_ALTERNATIVE_DATA: bool = True
    ENABLE_CACHING: bool = True
    ENABLE_COMPRESSION: bool = True
    ENABLE_PARALLEL_PROCESSING: bool = True
    ENABLE_GPU_ACCELERATION: bool = False
    
    # =============================================================================
    # SYSTEM RESOURCES
    # =============================================================================
    MAX_CPU_CORES: int = Field(4, ge=1, le=64)
    MAX_MEMORY_GB: int = Field(16, ge=1, le=256)
    MAX_DISK_GB: int = Field(100, ge=10, le=10000)
    ASYNC_WORKER_COUNT: int = Field(10, ge=1, le=100)
    THREAD_POOL_SIZE: int = Field(20, ge=1, le=200)
    MAX_CONCURRENT_REQUESTS: int = Field(100, ge=10, le=10000)
    REDIS_MAX_MEMORY: str = "2gb"
    CACHE_TTL_SECONDS: int = Field(300, ge=10, le=3600)
    
    @validator('ACTIVE_STRATEGIES')
    def validate_strategies(cls, v):
        """Validate active strategies list."""
        valid_strategies = {
            'momentum_breakout', 'mean_reversion', 'pairs_trading',
            'statistical_arbitrage', 'cross_asset_momentum', 'regime_switching'
        }
        strategies = [s.strip() for s in v.split(',')]
        for strategy in strategies:
            if strategy not in valid_strategies:
                raise ValueError(f"Invalid strategy: {strategy}")
        return v
    
    @validator('SIGNAL_TIMEFRAMES')
    def validate_timeframes(cls, v):
        """Validate signal timeframes."""
        valid_timeframes = {'1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w'}
        timeframes = [tf.strip() for tf in v.split(',')]
        for timeframe in timeframes:
            if timeframe not in valid_timeframes:
                raise ValueError(f"Invalid timeframe: {timeframe}")
        return v
    
    @validator('DATABASE_URL', 'TIMESCALEDB_URL')
    def validate_database_url(cls, v):
        """Validate database connection strings."""
        if not v.startswith(('postgresql://', 'sqlite:///')):
            raise ValueError("Database URL must use postgresql:// or sqlite:// scheme")
        return v
    
    @validator('REDIS_URL')
    def validate_redis_url(cls, v):
        """Validate Redis connection string."""
        if not v.startswith('redis://'):
            raise ValueError("Redis URL must use redis:// scheme")
        return v
    
    def get_active_strategies_list(self) -> List[str]:
        """Get list of active strategies."""
        return [s.strip() for s in self.ACTIVE_STRATEGIES.split(',')]
    
    def get_signal_timeframes_list(self) -> List[str]:
        """Get list of signal timeframes."""
        return [tf.strip() for tf in self.SIGNAL_TIMEFRAMES.split(',')]
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.ENVIRONMENT == Environment.PRODUCTION
    
    def is_live_trading(self) -> bool:
        """Check if live trading is enabled."""
        return self.TRADING_MODE == TradingMode.LIVE
    
    def get_exchange_config(self, exchange: str) -> Dict[str, Any]:
        """Get configuration for a specific exchange."""
        exchange_configs = {
            'binance': {
                'api_key': self.BINANCE_API_KEY,
                'secret_key': self.BINANCE_SECRET_KEY,
                'sandbox': self.BINANCE_SANDBOX,
                'base_url': self.BINANCE_BASE_URL
            },
            'coinbase': {
                'api_key': self.COINBASE_API_KEY,
                'secret_key': self.COINBASE_SECRET_KEY,
                'passphrase': self.COINBASE_PASSPHRASE,
                'sandbox': self.COINBASE_SANDBOX,
                'base_url': self.COINBASE_BASE_URL
            },
            'alpaca': {
                'api_key': self.ALPACA_API_KEY,
                'secret_key': self.ALPACA_SECRET_KEY,
                'base_url': self.ALPACA_BASE_URL
            }
        }
        return exchange_configs.get(exchange.lower(), {})
    
    def get_risk_limits(self) -> Dict[str, float]:
        """Get risk management limits as a dictionary."""
        return {
            'max_portfolio_drawdown': self.MAX_PORTFOLIO_DRAWDOWN,
            'max_asset_drawdown': self.MAX_ASSET_DRAWDOWN,
            'daily_loss_limit': self.DAILY_LOSS_LIMIT,
            'max_position_size': self.MAX_POSITION_SIZE,
            'max_sector_exposure': self.MAX_SECTOR_EXPOSURE,
            'max_crypto_leverage': self.MAX_CRYPTO_LEVERAGE,
            'max_equity_leverage': self.MAX_EQUITY_LEVERAGE
        }
    
    def get_technical_indicators_config(self) -> Dict[str, Any]:
        """Get technical indicators configuration."""
        return {
            'rsi_period': self.RSI_PERIOD,
            'macd_fast': self.MACD_FAST,
            'macd_slow': self.MACD_SLOW,
            'macd_signal': self.MACD_SIGNAL,
            'bb_period': self.BB_PERIOD,
            'bb_std': self.BB_STD,
            'atr_period': self.ATR_PERIOD
        }
    
    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True
        use_enum_values = True


# Cache settings instance to avoid re-reading environment variables
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are loaded once and reused.
    """
    return Settings()


# Convenience functions for common configuration needs
def get_database_url(test: bool = False) -> str:
    """Get appropriate database URL based on environment."""
    settings = get_settings()
    if test:
        return settings.TEST_DATABASE_URL
    elif settings.ENVIRONMENT == Environment.DEVELOPMENT:
        return settings.DEV_DATABASE_URL
    else:
        return settings.DATABASE_URL


def get_log_level() -> str:
    """Get current log level."""
    return get_settings().LOG_LEVEL.value


def is_debug_mode() -> bool:
    """Check if debug mode is enabled."""
    return get_settings().DEBUG_MODE


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a specific feature flag is enabled."""
    settings = get_settings()
    feature_map = {
        'rl_trading': settings.ENABLE_RL_TRADING,
        'nlp_sentiment': settings.ENABLE_NLP_SENTIMENT,
        'cross_asset_signals': settings.ENABLE_CROSS_ASSET_SIGNALS,
        'options_trading': settings.ENABLE_OPTIONS_TRADING,
        'futures_trading': settings.ENABLE_FUTURES_TRADING,
        'alternative_data': settings.ENABLE_ALTERNATIVE_DATA,
        'caching': settings.ENABLE_CACHING,
        'compression': settings.ENABLE_COMPRESSION,
        'parallel_processing': settings.ENABLE_PARALLEL_PROCESSING,
        'gpu_acceleration': settings.ENABLE_GPU_ACCELERATION
    }
    return feature_map.get(feature_name.lower(), False)


# Export main settings instance for convenience
settings = get_settings()
