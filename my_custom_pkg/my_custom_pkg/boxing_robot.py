#!/usr/bin/env python3
# coding=utf8
"""
Summon Robot Boxing Script
This script makes the robot play the boxing action once when executed.
"""

import rclpy
from rclpy.node import Node
from puppy_control_msgs.srv import SetRunActionName
import sys
import os
import subprocess
import time
import threading

# Add the puppypi_control path to access action group functions
sys.path.append('/home/ubuntu/software/puppypi_control')
from action_group_control import runActionGroup, stopActionGroup


class BoxingRobotNode(Node):
    def __init__(self):
        super().__init__('boxing_robot_node')
        
        # Create service client for running action groups
        self.run_action_group_srv = self.create_client(
            SetRunActionName, 
            '/puppy_control/runActionGroup'
        )
        
        # Try to start puppy_control if not running
        self.start_puppy_control_if_needed()
        
        # Wait for the service to be available with timeout
        timeout_seconds = 15
        start_time = self.get_clock().now()
        
        while not self.run_action_group_srv.wait_for_service(timeout_sec=1.0):
            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout_seconds:
                self.get_logger().error(f'Timeout waiting for runActionGroup service after {timeout_seconds} seconds.')
                self.get_logger().error('Please make sure the puppy_control node is running:')
                self.get_logger().error('  ros2 run puppy_control puppy_control')
                return
            self.get_logger().info(f'Waiting for runActionGroup service... ({elapsed:.1f}s elapsed)')
        
        self.get_logger().info('Boxing Robot Node initialized successfully!')
        
    def start_puppy_control_if_needed(self):
        """
        Check if puppy_control is running and start it if needed
        """
        try:
            # Check if puppy_control is already running
            result = subprocess.run(['ros2', 'node', 'list'], 
                                  capture_output=True, text=True, timeout=5)
            
            if 'puppy_control' in result.stdout:
                self.get_logger().info('puppy_control node is already running')
                return
            
            self.get_logger().info('puppy_control node not found, starting it...')
            
            # Start puppy_control in background
            def start_puppy_control():
                try:
                    subprocess.run(['ros2', 'run', 'puppy_control', 'puppy_control'], 
                                 timeout=30)  # Give it 30 seconds to start
                except subprocess.TimeoutExpired:
                    self.get_logger().info('puppy_control started successfully')
                except Exception as e:
                    self.get_logger().error(f'Failed to start puppy_control: {e}')
            
            # Start in a separate thread so it doesn't block
            thread = threading.Thread(target=start_puppy_control, daemon=True)
            thread.start()
            
            # Give it a moment to start
            time.sleep(2)
            
        except Exception as e:
            self.get_logger().warning(f'Could not check/start puppy_control: {e}')
            self.get_logger().warning('You may need to start puppy_control manually')
        
    def run_boxing_action(self):
        """
        Execute the boxing action once
        """
        try:
            self.get_logger().info('Starting boxing action...')
            
            # Run the boxing action group
            # You can choose between 'boxing.d6ac' or 'boxing2.d6ac'
            action_name = 'boxing.d6ac'
            runActionGroup(action_name, True)  # True means wait for completion
            
            self.get_logger().info('Boxing action completed successfully!')
            
        except Exception as e:
            self.get_logger().error(f'Error running boxing action: {e}')
            return False
        
        return True


def main(args=None):
    rclpy.init(args=args)
    
    # Create the boxing robot node
    boxing_node = BoxingRobotNode()
    
    try:
        # Execute the boxing action
        success = boxing_node.run_boxing_action()
        
        if success:
            boxing_node.get_logger().info('Boxing action executed successfully!')
        else:
            boxing_node.get_logger().error('Failed to execute boxing action!')
            
    except KeyboardInterrupt:
        boxing_node.get_logger().info('Boxing action interrupted by user')
    except Exception as e:
        boxing_node.get_logger().error(f'Unexpected error: {e}')
    finally:
        # Clean up
        try:
            boxing_node.destroy_node()
        except:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main() 