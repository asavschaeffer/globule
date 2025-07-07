#!/usr/bin/env python3
"""Test script to verify project structure."""

import sys
from pathlib import Path

def test_structure():
    """Test that all required files exist."""
    print("🧪 Testing Globule Project Structure")
    print("=" * 50)
    
    required_files = [
        "pyproject.toml",
        "globule/__init__.py",
        "globule/cli.py",
        "globule/config.py",
        "globule/storage.py",
        "globule/embedding_engine.py",
        "globule/parser_engine.py",
        "globule/query_engine.py",
        "globule/processor.py",
        "globule/synthesis.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✓ {file_path}")
    
    if missing_files:
        print(f"\n✗ Missing files: {missing_files}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 All required files exist!")
    
    # Test basic Python syntax
    print("\n📝 Testing Python syntax...")
    for file_path in required_files:
        if file_path.endswith('.py'):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Basic syntax check
                compile(content, file_path, 'exec')
                print(f"✓ {file_path} - syntax OK")
            except SyntaxError as e:
                print(f"✗ {file_path} - syntax error: {e}")
                return False
            except Exception as e:
                print(f"? {file_path} - could not check: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Project structure is complete!")
    print("\nTo install dependencies and test:")
    print("1. export PATH=\"/home/asas/.local/bin:$PATH\"")
    print("2. poetry install")
    print("3. poetry run globule config")
    print("4. poetry run globule add 'My first thought'")
    
    return True

if __name__ == "__main__":
    success = test_structure()
    sys.exit(0 if success else 1)