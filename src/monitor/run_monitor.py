#!/usr/bin/env python
"""
Script to run the Tensolr monitoring server
"""
import uvicorn
import sys
import os
from src.monitor.main import app

def main():
    print("Starting Tensolr Monitor Server...")
    print("Access the dashboard at: http://localhost:8000")
    
    uvicorn.run(
        "src.monitor.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload during development
    )

if __name__ == "__main__":
    main()