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

try:
    import uvicorn
    
    print("=" * 70)
    print("🚀 Starting Leasing Analyzer Web Server")
    print("=" * 70)
    print("📱 Open http://localhost:8000 in your browser")
    print("=" * 70)
    print("Press Ctrl+C to stop")
    print("=" * 70)
    
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
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

