#!/usr/bin/env python3
"""
Mirix - AI Assistant Application
Entry point for the Mirix application.
"""

import sys
import argparse
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point for Mirix application."""
    parser = argparse.ArgumentParser(description='Mirix AI Assistant Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=None, help='Port to bind the server to')
    
    args = parser.parse_args()
    
    # Determine port from command line, environment variable, or default
    port = args.port
    if port is None:
        port = int(os.environ.get('PORT', 8000))
    
    print(f"Starting Mirix server on {args.host}:{port}")
    
    import uvicorn
    from mirix.server import app
    uvicorn.run(app, host=args.host, port=port)

if __name__ == "__main__":
    main() 