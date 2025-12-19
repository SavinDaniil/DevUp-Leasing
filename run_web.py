#!/usr/bin/env python3
"""
Simple script to run the web server.
Usage: python run_web.py
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Set PYTHONPATH environment variable for uvicorn subprocess
# This ensures the reload subprocess can find the 'api' module
pythonpath = os.environ.get("PYTHONPATH", "")
if pythonpath:
    os.environ["PYTHONPATH"] = f"{current_dir}{os.pathsep}{pythonpath}"
else:
    os.environ["PYTHONPATH"] = current_dir

if __name__ == "__main__":
    try:
        import uvicorn
        
        print("=" * 70)
        print("🚀 Starting Leasing Analyzer Web Server")
        print("=" * 70)
        print("📱 Open http://localhost:8000 in your browser")
        print("=" * 70)
        print("Press Ctrl+C to stop")
        print("=" * 70)
        
        # Увеличиваем таймауты для долгих запросов
        uvicorn.run(
            "api.main:app", 
            host="127.0.0.1", 
            port=8000, 
            reload=True,
            timeout_keep_alive=1200,  # 20 минут для keep-alive
            timeout_graceful_shutdown=30  # 30 секунд для graceful shutdown
        )
    except ImportError:
        print("❌ Error: uvicorn not installed")
        print("   Run: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)