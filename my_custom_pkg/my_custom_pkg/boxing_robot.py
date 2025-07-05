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
        
        # Wait for the service to be available
        while not self.run_action_group_srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for runActionGroup service...')
        
        self.get_logger().info('Boxing Robot Node initialized successfully!')
        
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
        boxing_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 