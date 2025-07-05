#!/usr/bin/env python3
# coding=utf8
"""
Simple Boxing Robot Launcher
This script can be run directly to make the robot play the boxing action once.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the boxing robot module
from my_custom_pkg.my_custom_pkg.boxing_robot import main

if __name__ == '__main__':
    print("Starting Boxing Robot...")
    main() 