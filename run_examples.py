#!/usr/bin/env python3
"""
Setup script to run examples from the project root directory
"""

import subprocess
import sys
import os

def run_example(example_name):
    """Run a specific example by name"""
    example_path = f"examples/{example_name}.py"
    
    if not os.path.exists(example_path):
        print(f"Example {example_name} not found at {example_path}")
        return False
    
    print(f"Running {example_path}...")
    # Run with PYTHONPATH set to include project root
    env = os.environ.copy()
    project_root = os.path.dirname(os.path.abspath(__file__))
    env['PYTHONPATH'] = project_root
    
    result = subprocess.run([sys.executable, "-m", "examples." + example_name], 
                          cwd=project_root,
                          env=env)
    
    # If the -m approach doesn't work (because examples isn't a package), try the direct approach
    if result.returncode != 0:
        print(f"Module approach failed, trying direct approach...")
        result = subprocess.run([sys.executable, example_path], 
                              cwd=project_root,
                              env=env)
    
    return result.returncode == 0

def list_examples():
    """List all available examples"""
    examples_dir = "examples"
    examples = [f for f in os.listdir(examples_dir) if f.endswith('.py')]
    print("Available examples:")
    for example in examples:
        print(f"  - {example[:-3]}")  # Remove .py extension

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_examples.py <example_name> | list")
        print("  example_name: Name of the example to run (without .py)")
        print("  list: List all available examples")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "list":
        list_examples()
    else:
        success = run_example(command)
        if success:
            print(f"Example {command} ran successfully!")
        else:
            print(f"Example {command} failed!")
            sys.exit(1)