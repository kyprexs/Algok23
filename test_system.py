#!/usr/bin/env python3
"""
AgloK23 System Health Check and Component Test
"""

import sys
import os
import asyncio
import importlib
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_imports():
    """Test if core Python packages can be imported"""
    print("🔍 Testing Core Imports...")
    
    import_tests = {
        "fastapi": "Web framework",
        "uvicorn": "ASGI server",
        "pydantic": "Data validation",
        "pandas": "Data processing",
        "numpy": "Numerical computing",
        "redis": "Redis client",
        "sqlalchemy": "Database ORM",
        "asyncio": "Async programming",
        "websockets": "WebSocket client",
        "aiohttp": "Async HTTP client",
        "matplotlib": "Plotting",
        "plotly": "Interactive plots",
        "sklearn": "Machine learning",
    }
    
    results = {}
    for module_name, description in import_tests.items():
        try:
            importlib.import_module(module_name)
            results[module_name] = "✅ OK"
        except ImportError as e:
            results[module_name] = f"❌ Failed: {str(e)[:50]}"
        except Exception as e:
            results[module_name] = f"⚠️  Error: {str(e)[:50]}"
    
    print("\nImport Test Results:")
    for module, status in results.items():
        print(f"  {module:20} {status}")
    
    return results

def test_advanced_imports():
    """Test advanced ML and trading specific imports"""
    print("\n🔍 Testing Advanced Imports...")
    
    advanced_tests = {
        "mlflow": "ML experiment tracking",
        "tensorflow": "Deep learning",
        "torch": "PyTorch",
        "xgboost": "Gradient boosting",
        "lightgbm": "LightGBM",
    }
    
    results = {}
    for module_name, description in advanced_tests.items():
        try:
            importlib.import_module(module_name)
            results[module_name] = "✅ OK"
        except ImportError as e:
            results[module_name] = f"❌ Not installed"
        except Exception as e:
            results[module_name] = f"⚠️  Error: {str(e)[:50]}"
    
    print("\nAdvanced Import Results:")
    for module, status in results.items():
        print(f"  {module:20} {status}")
    
    return results

def test_project_structure():
    """Test if project structure exists"""
    print("\n🔍 Testing Project Structure...")
    
    required_paths = [
        "src",
        "src/config",
        "src/core",
        "src/core/data",
        "src/core/features",
        "src/core/models",
        "src/core/strategy",
        "src/core/risk",
        "src/core/execution",
        "src/core/monitoring",
        "config",
        "scripts",
        "notebooks",
    ]
    
    results = {}
    for path in required_paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                results[path] = "✅ Directory exists"
            else:
                results[path] = "✅ File exists"
        else:
            results[path] = "❌ Missing"
    
    print("\nProject Structure Results:")
    for path, status in results.items():
        print(f"  {path:30} {status}")
    
    return results

def test_config_loading():
    """Test configuration loading"""
    print("\n🔍 Testing Configuration Loading...")
    
    try:
        from config.settings import Settings
        settings = Settings()
        print("✅ Settings loaded successfully")
        print(f"   Environment: {settings.ENVIRONMENT}")
        print(f"   Log Level: {settings.LOG_LEVEL}")
        return {"config": "✅ OK"}
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return {"config": f"❌ Failed: {e}"}

def test_data_models():
    """Test data model imports"""
    print("\n🔍 Testing Data Models...")
    
    try:
        from core.data.models import MarketData, OHLCV, Order, Position, Portfolio, Signal
        print("✅ Data models imported successfully")
        
        # Test model instantiation
        from datetime import datetime
        test_ohlcv = OHLCV(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            source="test",
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=1000.0
        )
        print("✅ OHLCV model instantiation works")
        
        return {"data_models": "✅ OK"}
    except Exception as e:
        print(f"❌ Data models test failed: {e}")
        return {"data_models": f"❌ Failed: {e}"}

def test_feature_engine():
    """Test feature engine"""
    print("\n🔍 Testing Feature Engine...")
    
    try:
        from core.features.engine import FeatureEngine
        engine = FeatureEngine()
        print("✅ Feature engine created successfully")
        return {"feature_engine": "✅ OK"}
    except Exception as e:
        print(f"❌ Feature engine test failed: {e}")
        return {"feature_engine": f"❌ Failed: {e}"}

async def test_async_functionality():
    """Test async functionality"""
    print("\n🔍 Testing Async Functionality...")
    
    try:
        # Simple async test
        await asyncio.sleep(0.1)
        print("✅ Basic async/await works")
        
        # Test async http client
        import aiohttp
        async with aiohttp.ClientSession() as session:
            # Test with a reliable endpoint
            try:
                async with session.get('https://httpbin.org/status/200', timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        print("✅ Async HTTP client works")
                        return {"async": "✅ OK"}
                    else:
                        print(f"⚠️  HTTP returned status: {response.status}")
                        return {"async": "⚠️  HTTP issues"}
            except asyncio.TimeoutError:
                print("⚠️  HTTP timeout - network may be limited")
                return {"async": "⚠️  Network timeout"}
            except Exception as e:
                print(f"⚠️  HTTP test failed: {e}")
                return {"async": "⚠️  HTTP failed"}
                
    except Exception as e:
        print(f"❌ Async test failed: {e}")
        return {"async": f"❌ Failed: {e}"}

def test_database_connections():
    """Test database connection capabilities (without actual connections)"""
    print("\n🔍 Testing Database Connection Libraries...")
    
    results = {}
    
    # Test SQLAlchemy
    try:
        from sqlalchemy import create_engine, text
        # Create in-memory SQLite for testing
        engine = create_engine("sqlite:///:memory:", echo=False)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            if row[0] == 1:
                print("✅ SQLAlchemy + SQLite works")
                results["sqlalchemy"] = "✅ OK"
            else:
                results["sqlalchemy"] = "❌ Query failed"
    except Exception as e:
        print(f"❌ SQLAlchemy test failed: {e}")
        results["sqlalchemy"] = f"❌ Failed: {e}"
    
    return results

async def run_all_tests():
    """Run all system tests"""
    print("=" * 60)
    print("🚀 AgloK23 System Health Check")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    all_results = {}
    
    # Core tests
    all_results.update(test_core_imports())
    all_results.update(test_advanced_imports())
    all_results.update(test_project_structure())
    all_results.update(test_config_loading())
    all_results.update(test_data_models())
    all_results.update(test_feature_engine())
    all_results.update(await test_async_functionality())
    all_results.update(test_database_connections())
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for v in all_results.values() if v.startswith("✅"))
    warnings = sum(1 for v in all_results.values() if v.startswith("⚠️"))
    failed = sum(1 for v in all_results.values() if v.startswith("❌"))
    
    print(f"✅ Passed:   {passed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"❌ Failed:   {failed}")
    print(f"📈 Success:  {passed / len(all_results) * 100:.1f}%")
    
    if failed == 0 and warnings <= 2:
        print("\n🎉 System is ready for development!")
    elif failed <= 2:
        print("\n⚠️  System has minor issues but should work")
    else:
        print("\n❌ System has significant issues that need attention")
    
    return all_results

if __name__ == "__main__":
    try:
        results = asyncio.run(run_all_tests())
        
        # Write results to file
        with open("system_health_report.txt", "w", encoding='utf-8') as f:
            f.write(f"AgloK23 System Health Report - {datetime.now()}\n")
            f.write("=" * 50 + "\n\n")
            for test, result in results.items():
                # Replace unicode characters for file output
                clean_result = result.replace('✅', 'OK').replace('❌', 'FAILED').replace('⚠️', 'WARNING')
                f.write(f"{test}: {clean_result}\n")
        
        print(f"\n📄 Detailed report saved to: system_health_report.txt")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test runner failed: {e}")
        sys.exit(1)
