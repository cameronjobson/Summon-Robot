#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import random
import sys
import traceback

# Custom implementation to avoid external tf transformations dependency
def quaternion_from_euler(roll: float, pitch: float, yaw: float):
    """
    Convert Euler angles (roll, pitch, yaw) to quaternion (x, y, z, w).
    """
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return [x, y, z, w]

class RandomNavigator(Node):
    def __init__(self):
        super().__init__('random_navigator')
        # Declare parameters with defaults
        self.declare_parameter('top_left', [0.0, 0.0])
        self.declare_parameter('top_right', [2.0, 0.0])
        self.declare_parameter('bottom_right', [2.0, 2.0])
        self.declare_parameter('bottom_left', [0.0, 2.0])
        self.declare_parameter('start_corner', 'top_left')
        self.declare_parameter('start_yaw', 0.0)
        self.declare_parameter('amcl_wait_time', 2.0)
        self.declare_parameter('num_random_goals', 10)
        self.declare_parameter('max_speed', 0.3)
        self.declare_parameter('max_accel', 0.2)

        # Read parameters
        self.corners = {
            'top_left': self.get_parameter('top_left').get_parameter_value().double_array_value,
            'top_right': self.get_parameter('top_right').get_parameter_value().double_array_value,
            'bottom_right': self.get_parameter('bottom_right').get_parameter_value().double_array_value,
            'bottom_left': self.get_parameter('bottom_left').get_parameter_value().double_array_value,
        }
        self.start_corner = self.get_parameter('start_corner').get_parameter_value().string_value
        self.start_yaw = self.get_parameter('start_yaw').get_parameter_value().double_value
        self.amcl_wait_time = self.get_parameter('amcl_wait_time').get_parameter_value().double_value
        self.num_random_goals = self.get_parameter('num_random_goals').get_parameter_value().integer_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.max_accel = self.get_parameter('max_accel').get_parameter_value().double_value

        # Validate corners
        for name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if len(self.corners[name]) != 2:
                self.get_logger().error(f"Corner {name} must be a list of two floats [x, y].")
                sys.exit(1)

        # Set up Nav2 Simple Commander
        self.navigator = BasicNavigator()
        print('Available BasicNavigator methods:', dir(self.navigator))
        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Nav2 is active.')

        # Set initial pose
        self.set_initial_pose()
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Waiting {self.amcl_wait_time} seconds for AMCL convergence...")
        # Use time.sleep if needed
        import time; time.sleep(self.amcl_wait_time)

        # Calibration lap
        self.calibration_lap()

        # Random goal loop
        self.random_goal_loop()

    def set_initial_pose(self):
        pose = self.corner_to_pose(self.start_corner, self.start_yaw)
        self.navigator.setInitialPose(pose)
        self.get_logger().info(
            f"Initial pose set to {self.start_corner} at {pose.pose.position.x}, {pose.pose.position.y}, yaw={self.start_yaw}"
        )

    def corner_to_pose(self, corner_name: str, yaw: float = 0.0) -> PoseStamped:
        x, y = self.corners[corner_name]
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = self.navigator.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]
        return pose

    def calibration_lap(self):
        self.get_logger().info('Starting calibration lap...')
        order = ['top_left', 'top_right', 'bottom_right', 'bottom_left', 'top_left']
        for i in range(len(order) - 1):
            start = order[i]
            goal = order[i + 1]
            pose = self.corner_to_pose(goal, 0.0)
            self.get_logger().info(
                f"Navigating from {start} to {goal} at {pose.pose.position.x}, {pose.pose.position.y}"
            )
            try:
                self.navigator.goToPose(pose)
                result = self.navigator.waitUntilNavArrived()
                if result:
                    self.get_logger().info(f"Reached {goal} successfully.")
                else:
                    self.get_logger().warn(f"Failed to reach {goal}.")
            except Exception as e:
                self.get_logger().error(
                    f"Exception during navigation: {e}\n{traceback.format_exc()}"
                )

    def random_goal_loop(self):
        self.get_logger().info('Starting random goal loop...')
        corners = self.corners
        triangles = [
            (corners['top_left'], corners['top_right'], corners['bottom_left']),
            (corners['top_right'], corners['bottom_right'], corners['bottom_left'])
        ]
        count = 0
        try:
            while self.num_random_goals <= 0 or count < self.num_random_goals:
                # Pick triangle
                tri_idx = random.randint(0, 1)
                A, B, C = triangles[tri_idx]
                u, v = random.random(), random.random()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
                y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.navigator.get_clock().now().to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                # Random yaw
                yaw = random.uniform(-math.pi, math.pi)
                q = quaternion_from_euler(0.0, 0.0, yaw)
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                self.get_logger().info(
                    f"Navigating to random goal {count+1}: ({x:.2f}, {y:.2f}, yaw={yaw:.2f})"
                )
                try:
                    self.navigator.goToPose(pose)
                    result = self.navigator.waitUntilNavArrived()
                    if result:
                        self.get_logger().info(f"Random goal {count+1} reached successfully.")
                    else:
                        self.get_logger().warn(f"Random goal {count+1} failed.")
                except Exception as e:
                    self.get_logger().error(
                        f"Exception during random navigation: {e}\n{traceback.format_exc()}"
                    )
                count += 1
        except KeyboardInterrupt:
            self.get_logger().info('Shutting down on user request.')
            rclpy.shutdown()
        self.get_logger().info('Random goal loop complete.')
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    try:
        RandomNavigator()
    except Exception as e:
        print(f"Exception in main: {e}\n{traceback.format_exc()}")
        rclpy.shutdown()

if __name__ == '__main__':
    main()
