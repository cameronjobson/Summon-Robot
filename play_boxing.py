#!/usr/bin/env python3
# encoding: utf-8

"""
Summon Robot Boxing Demo Launcher
This script plays the Boxing action on the robot once when started.
"""

import sys
import os

# Add the my_custom_pkg to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'my_custom_pkg', 'my_custom_pkg'))

# Import and run the boxing demo
from boxing_demo import main

if __name__ == "__main__":
    main() 