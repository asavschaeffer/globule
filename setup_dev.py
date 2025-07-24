#!/usr/bin/env python3
"""
Development setup script for Globule MVP.

This script sets up the development environment and pulls required models.
"""

import asyncio
import subprocess
import sys
import os
from pathlib import Path

async def run_command(cmd: list, cwd: Path = None):
    """Run a command and return success status"""
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"✅ {' '.join(cmd)}")
            return True
        else:
            print(f"❌ {' '.join(cmd)}")
            if stderr:
                print(f"   Error: {stderr.decode()}")
            return False
    except Exception as e:
        print(f"❌ {' '.join(cmd)} - {e}")
        return False

async def main():
    """Main setup routine"""
    print("🚀 Setting up Globule development environment...")
    
    project_root = Path(__file__).parent
    
    # 1. Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    success = await run_command([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ], cwd=project_root)
    
    if not success:
        print("Failed to install dependencies. Exiting.")
        return False
    
    # 2. Install package in development mode
    print("\n🔧 Installing Globule in development mode...")
    success = await run_command([
        sys.executable, "-m", "pip", "install", "-e", "."
    ], cwd=project_root)
    
    if not success:
        print("Failed to install Globule package. Exiting.")
        return False
    
    # 3. Start Ollama container
    print("\n🐳 Starting Ollama container...")
    success = await run_command([
        "docker-compose", "up", "-d"
    ], cwd=project_root)
    
    if not success:
        print("Failed to start Ollama. Please ensure Docker is installed and running.")
        return False
    
    # 4. Wait for Ollama to be ready
    print("\n⏳ Waiting for Ollama to be ready...")
    await asyncio.sleep(10)
    
    # 5. Pull required models
    print("\n🤖 Pulling required AI models...")
    
    models = ["mxbai-embed-large", "llama3.2:3b"]
    for model in models:
        print(f"   Pulling {model}...")
        success = await run_command([
            "docker", "exec", "globule-ollama", "ollama", "pull", model
        ])
        if not success:
            print(f"   ⚠️  Failed to pull {model}. You can pull it manually later.")
    
    # 6. Test installation
    print("\n🧪 Testing installation...")
    
    # Test CLI is available
    success = await run_command([sys.executable, "-c", "from globule.cli import main; print('CLI import successful')"])
    if not success:
        print("CLI import failed")
        return False
    
    print("\n✅ Setup complete!")
    print("\n📋 Next steps:")
    print("   1. Try: globule add \"This is my first thought\"")
    print("   2. Try: globule draft")
    print("   3. Check the documentation in docs/ for more details")
    
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)