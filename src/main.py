"""
AgloK23 Advanced Quantitative Trading System
============================================

Main application entry point that orchestrates the entire trading system including:
- Data ingestion and processing
- Feature engineering pipeline  
- ML model inference
- Strategy execution
- Risk management
- Monitoring and alerting

Author: AgloK23 Team
License: MIT
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import Settings, get_settings
from src.data.ingestion_manager import DataIngestionManager
from src.features.feature_engine import FeatureEngine
from src.models.model_manager import ModelManager
from src.strategies.strategy_engine import StrategyEngine
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.monitoring.metrics_collector import MetricsCollector
from src.monitoring.health_monitor import HealthMonitor

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global system components
system_components: Dict[str, Any] = {}
shutdown_event = asyncio.Event()


class TradingSystemOrchestrator:
    """
    Main orchestrator for the AgloK23 trading system.
    
    Manages lifecycle of all system components and coordinates
    data flow between different services.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.components = {}
        self.running = False
        self.health_monitor = None
        
    async def initialize_system(self):
        """Initialize all system components in proper dependency order."""
        logger.info("🚀 Initializing AgloK23 Trading System...")
        
        try:
            # 1. Health Monitoring (needs to start first)
            logger.info("📊 Starting health monitor...")
            self.health_monitor = HealthMonitor(self.settings)
            await self.health_monitor.start()
            self.components['health_monitor'] = self.health_monitor
            
            # 2. Metrics Collection
            logger.info("📈 Starting metrics collector...")
            metrics_collector = MetricsCollector(self.settings)
            await metrics_collector.start()
            self.components['metrics_collector'] = metrics_collector
            
            # 3. Data Ingestion Manager
            logger.info("📡 Starting data ingestion manager...")
            data_manager = DataIngestionManager(self.settings)
            await data_manager.start()
            self.components['data_manager'] = data_manager
            
            # 4. Feature Engineering Engine
            logger.info("🔧 Starting feature engineering engine...")
            feature_engine = FeatureEngine(self.settings, data_manager)
            await feature_engine.start()
            self.components['feature_engine'] = feature_engine
            
            # 5. ML Model Manager
            logger.info("🧠 Loading ML models...")
            model_manager = ModelManager(self.settings, feature_engine)
            await model_manager.load_models()
            self.components['model_manager'] = model_manager
            
            # 6. Risk Manager (critical - must start before strategies)
            logger.info("🛡️ Starting risk manager...")
            risk_manager = RiskManager(self.settings)
            await risk_manager.start()
            self.components['risk_manager'] = risk_manager
            
            # 7. Order Manager
            logger.info("📋 Starting order manager...")
            order_manager = OrderManager(self.settings, risk_manager)
            await order_manager.start()
            self.components['order_manager'] = order_manager
            
            # 8. Strategy Engine (depends on all above components)
            logger.info("🎯 Starting strategy engine...")
            strategy_engine = StrategyEngine(
                settings=self.settings,
                model_manager=model_manager,
                risk_manager=risk_manager,
                order_manager=order_manager
            )
            await strategy_engine.start()
            self.components['strategy_engine'] = strategy_engine
            
            # Update global components reference
            global system_components
            system_components.update(self.components)
            
            logger.info("✅ AgloK23 Trading System initialized successfully!")
            self.running = True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize trading system: {e}")
            await self.shutdown_system()
            raise
    
    async def run_system(self):
        """Main system execution loop."""
        logger.info("🏃‍♂️ Starting main trading loop...")
        
        try:
            while self.running and not shutdown_event.is_set():
                # Main trading loop - orchestrate all components
                await self._execute_trading_cycle()
                
                # Brief sleep to prevent CPU overload
                await asyncio.sleep(0.1)  # 100ms cycle time
                
        except Exception as e:
            logger.error(f"💥 Error in main trading loop: {e}")
            raise
        finally:
            logger.info("🛑 Main trading loop stopped")
    
    async def _execute_trading_cycle(self):
        """Execute one complete trading cycle."""
        try:
            # 1. Check system health
            if not await self.health_monitor.check_system_health():
                logger.warning("⚠️ System health check failed - pausing trading")
                return
            
            # 2. Process any new market data
            # (Data ingestion runs in background, just check for new data)
            
            # 3. Update features if new data available
            # (Feature engine processes data streams automatically)
            
            # 4. Generate trading signals
            strategy_engine = self.components.get('strategy_engine')
            if strategy_engine:
                await strategy_engine.generate_signals()
            
            # 5. Execute trades (if any signals generated)
            order_manager = self.components.get('order_manager')
            if order_manager:
                await order_manager.process_pending_orders()
            
            # 6. Update risk metrics
            risk_manager = self.components.get('risk_manager')
            if risk_manager:
                await risk_manager.update_portfolio_risk()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            # Continue running but log the error
    
    async def shutdown_system(self):
        """Gracefully shutdown all system components."""
        logger.info("🛑 Shutting down AgloK23 Trading System...")
        self.running = False
        
        # Shutdown in reverse dependency order
        shutdown_order = [
            'strategy_engine',
            'order_manager', 
            'risk_manager',
            'model_manager',
            'feature_engine',
            'data_manager',
            'metrics_collector',
            'health_monitor'
        ]
        
        for component_name in shutdown_order:
            if component_name in self.components:
                try:
                    logger.info(f"Shutting down {component_name}...")
                    component = self.components[component_name]
                    if hasattr(component, 'stop'):
                        await component.stop()
                    elif hasattr(component, 'shutdown'):
                        await component.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {component_name}: {e}")
        
        logger.info("✅ AgloK23 Trading System shutdown complete")


# FastAPI application with lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    settings = get_settings()
    orchestrator = TradingSystemOrchestrator(settings)
    
    # Startup
    try:
        await orchestrator.initialize_system()
        # Start the main trading loop in background
        trading_task = asyncio.create_task(orchestrator.run_system())
        
        yield  # Application is running
        
    finally:
        # Shutdown
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass
        await orchestrator.shutdown_system()


# Create FastAPI application
app = FastAPI(
    title="AgloK23 Advanced Quantitative Trading System",
    description="Institutional-grade algorithmic trading platform with ML-enhanced strategies",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Routes
@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "system": "AgloK23 Advanced Quantitative Trading System",
        "version": "1.0.0",
        "status": "running" if system_components else "initializing",
        "components": list(system_components.keys()) if system_components else []
    }


@app.get("/health")
async def health_check():
    """System health check endpoint."""
    if 'health_monitor' in system_components:
        health_status = await system_components['health_monitor'].get_health_status()
        return health_status
    else:
        return JSONResponse(
            status_code=503,
            content={"status": "initializing", "message": "System still starting up"}
        )


@app.get("/metrics")
async def get_metrics():
    """Get current system metrics."""
    if 'metrics_collector' in system_components:
        metrics = await system_components['metrics_collector'].get_current_metrics()
        return metrics
    else:
        raise HTTPException(status_code=503, detail="Metrics not available yet")


@app.get("/portfolio")
async def get_portfolio():
    """Get current portfolio status."""
    if 'risk_manager' in system_components:
        portfolio = await system_components['risk_manager'].get_portfolio_status()
        return portfolio
    else:
        raise HTTPException(status_code=503, detail="Portfolio data not available yet")


@app.get("/positions")
async def get_positions():
    """Get current trading positions."""
    if 'order_manager' in system_components:
        positions = await system_components['order_manager'].get_current_positions()
        return positions
    else:
        raise HTTPException(status_code=503, detail="Position data not available yet")


@app.get("/strategies")
async def get_strategies():
    """Get active trading strategies and their status."""
    if 'strategy_engine' in system_components:
        strategies = await system_components['strategy_engine'].get_strategy_status()
        return strategies
    else:
        raise HTTPException(status_code=503, detail="Strategy data not available yet")


@app.post("/emergency-stop")
async def emergency_stop():
    """Emergency stop - close all positions and halt trading."""
    if 'risk_manager' in system_components:
        await system_components['risk_manager'].emergency_stop()
        return {"message": "Emergency stop activated - all trading halted"}
    else:
        raise HTTPException(status_code=503, detail="Risk manager not available")


@app.post("/shutdown")
async def shutdown_system():
    """Gracefully shutdown the trading system."""
    shutdown_event.set()
    return {"message": "System shutdown initiated"}


# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_event.set()


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


if __name__ == "__main__":
    """Main entry point for the trading system."""
    settings = get_settings()
    
    logger.info("🚀 Starting AgloK23 Advanced Quantitative Trading System")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Trading Mode: {settings.TRADING_MODE}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    
    try:
        # Run with uvicorn
        uvicorn.run(
            "src.main:app",
            host="0.0.0.0",
            port=8000,
            log_level=settings.LOG_LEVEL.lower(),
            reload=settings.ENVIRONMENT == "development",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("👋 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
