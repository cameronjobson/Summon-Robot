#!/usr/bin/env python3
# coding=utf8
"""
Boxing Robot Launcher
This script automatically starts puppy_control if needed and then runs the boxing action.
"""

import subprocess
import time
import sys
import os

def check_puppy_control_running():
    """Check if puppy_control node is already running"""
    try:
        result = subprocess.run(['ros2', 'node', 'list'], 
                              capture_output=True, text=True, timeout=5)
        return 'puppy_control' in result.stdout
    except Exception:
        return False

def start_puppy_control():
    """Start the puppy_control node"""
    print("Starting puppy_control node...")
    try:
        # Start puppy_control in background
        process = subprocess.Popen(['ros2', 'run', 'puppy_control', 'puppy_control'],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it time to start
        time.sleep(3)
        
        # Check if it started successfully
        if process.poll() is None:  # Still running
            print("puppy_control started successfully")
            return process
        else:
            print("Failed to start puppy_control")
            return None
    except Exception as e:
        print(f"Error starting puppy_control: {e}")
        return None

def run_boxing_script():
    """Run the boxing robot script"""
    print("Running boxing robot script...")
    try:
        # Change to the workspace directory
        workspace_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(workspace_dir)
        
        # Run the boxing script
        result = subprocess.run(['ros2', 'run', 'my_custom_pkg', 'boxing_robot'],
                              timeout=60)  # 60 second timeout
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("Boxing script timed out")
        return False
    except Exception as e:
        print(f"Error running boxing script: {e}")
        return False

def main():
    print("=== Boxing Robot Launcher ===")
    
    # Check if puppy_control is already running
    if check_puppy_control_running():
        print("puppy_control is already running")
        puppy_process = None
    else:
        # Start puppy_control
        puppy_process = start_puppy_control()
        if puppy_process is None:
            print("Failed to start puppy_control. Please start it manually:")
            print("  ros2 run puppy_control puppy_control")
            return 1
    
    try:
        # Wait a bit for services to be available
        print("Waiting for services to be ready...")
        time.sleep(2)
        
        # Run the boxing script
        success = run_boxing_script()
        
        if success:
            print("Boxing action completed successfully!")
        else:
            print("Boxing action failed!")
            return 1
            
    finally:
        # Clean up puppy_control if we started it
        if puppy_process:
            print("Stopping puppy_control...")
            try:
                puppy_process.terminate()
                puppy_process.wait(timeout=5)
            except:
                puppy_process.kill()
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 