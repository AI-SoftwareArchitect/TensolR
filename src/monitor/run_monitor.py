#!/usr/bin/env python
"""
Script to run the Tensolr monitoring server
"""
import uvicorn
import sys
import os
import argparse
from src.monitor.main import app

def main():
    parser = argparse.ArgumentParser(description='Tensolr Monitor Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()
    
    print(f"Starting Tensolr Monitor Server on {args.host}:{args.port}...")
    print(f"Access the dashboard at: http://{args.host.replace('0.0.0.0', 'localhost')}:{args.port}")
    
    uvicorn.run(
        "src.monitor.main:app",
        host=args.host,
        port=args.port,
        reload=False  # Disable auto-reload to prevent restarts
    )

if __name__ == "__main__":
    main()