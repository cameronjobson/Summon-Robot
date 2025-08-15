#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import random
import sys
import traceback
from typing import List, Tuple, Optional

# --- Math & quaternion helpers ---
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

def yaw_towards(from_x: float, from_y: float, to_x: float, to_y: float) -> float:
    """Heading (yaw) from (from_x, from_y) toward (to_x, to_y)."""
    return math.atan2(to_y - from_y, to_x - from_x)

def normalize_angle(theta: float) -> float:
    """Wrap angle to [-pi, pi]."""
    while theta > math.pi:
        theta -= 2.0 * math.pi
    while theta < -math.pi:
        theta += 2.0 * math.pi
    return theta

def yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    """Extract yaw from quaternion (robust general formula)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)

# --- Geofence helpers for a convex quad in CCW order ---
def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by

def norm(ax: float, ay: float) -> float:
    return math.hypot(ax, ay)

def push_inside_convex_quad(x: float, y: float, verts_ccw: List[Tuple[float, float]],
                            margin: float = 0.15, iters: int = 3) -> Tuple[float, float]:
    """
    Ensure (x,y) lies at least `margin` inside a convex quad (verts CCW).
    For each edge, if the point is closer than margin to the outside boundary,
    push inward along the inward normal. Iterate a few times.
    """
    px, py = x, y
    for _ in range(iters):
        moved = False
        for i in range(4):
            x1, y1 = verts_ccw[i]
            x2, y2 = verts_ccw[(i + 1) % 4]
            ex, ey = (x2 - x1), (y2 - y1)
            # inward normal for CCW polygon is the LEFT normal: (-ey, ex)
            nx, ny = (-ey, ex)
            nlen = norm(nx, ny)
            if nlen == 0.0:
                continue
            nx, ny = nx / nlen, ny / nlen
            # signed distance from point to edge along inward normal
            d = dot(px - x1, py - y1, nx, ny)
            if d < margin:
                delta = (margin - d)
                px += nx * delta
                py += ny * delta
                moved = True
        if not moved:
            break
    return px, py

def point_outside_convex_quad(x: float, y: float, verts_ccw: List[Tuple[float, float]],
                              tol: float = 0.0) -> bool:
    """
    Return True if (x,y) is outside by more than -tol.
    tol >= 0 allows a tiny bleed outside before triggering.
    """
    for i in range(4):
        x1, y1 = verts_ccw[i]
        x2, y2 = verts_ccw[(i + 1) % 4]
        ex, ey = (x2 - x1), (y2 - y1)
        nx, ny = (-ey, ex)  # inward normal
        nlen = norm(nx, ny)
        if nlen == 0.0:
            continue
        nx, ny = nx / nlen, ny / nlen
        d = dot(x - x1, y - y1, nx, ny)
        if d < -tol:
            return True
    return False

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
        # Smart-yaw feature toggles/thresholds
        self.declare_parameter('use_goal_heading', True)
        self.declare_parameter('close_goal_distance', 0.35)   # m
        self.declare_parameter('lookahead_distance', 0.40)     # m
        # Geofence controls
        self.declare_parameter('safety_margin', 0.15)          # m inside the fence
        self.declare_parameter('oob_recovery_enabled', True)   # auto-correct if outside
        self.declare_parameter('oob_check_period', 1.0)        # seconds between checks

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
        self.use_goal_heading = self.get_parameter('use_goal_heading').get_parameter_value().bool_value
        self.close_goal_distance = self.get_parameter('close_goal_distance').get_parameter_value().double_value
        self.lookahead_distance = self.get_parameter('lookahead_distance').get_parameter_value().double_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value
        self.oob_recovery_enabled = self.get_parameter('oob_recovery_enabled').get_parameter_value().bool_value
        self.oob_check_period = self.get_parameter('oob_check_period').get_parameter_value().double_value

        # Validate corners
        for name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if len(self.corners[name]) != 2:
                self.get_logger().error(f"Corner {name} must be a list of two floats [x, y].")
                sys.exit(1)

        # Define quad in CCW order (must match your actual geometry)
        self.quad_ccw: List[Tuple[float, float]] = [
            tuple(self.corners['top_left']),
            tuple(self.corners['top_right']),
            tuple(self.corners['bottom_right']),
            tuple(self.corners['bottom_left']),
        ]

        # OOB logic starts disabled; enabled after center hop
        self.oob_active: bool = False

        # Set up Nav2 Simple Commander
        self.navigator = BasicNavigator()
        print('Available BasicNavigator methods:', dir(self.navigator))
        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Nav2 is active.')

        # Try to apply speed limits (if supported by this version)
        try:
            self.navigator.setSpeedLimits(self.max_speed, self.max_accel, 0.0)
            self.get_logger().info(f"Speed limits set: max_speed={self.max_speed}, max_accel={self.max_accel}")
        except Exception as e:
            self.get_logger().warn(f"setSpeedLimits not supported or failed: {e}")

        # Set initial pose (preserved position/yaw; stamp at time=0 to use latest TF)
        self.set_initial_pose()
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Waiting {self.amcl_wait_time} seconds for AMCL convergence...")
        import time; time.sleep(self.amcl_wait_time)

        # Calibration lap (no OOB checks during this phase)
        self.calibration_lap()

        # Go to center; once reached, enable OOB logic
        self.go_to_center_and_enable_oob()

        # Random goal loop (OOB logic is now active)
        self.random_goal_loop()

    def set_initial_pose(self):
        pose = self.corner_to_pose(self.start_corner, self.start_yaw)
        # Let AMCL consume with latest TF to avoid future-time extrapolation warnings
        pose.header.stamp = rclpy.time.Time().to_msg()  # time=0 special stamp
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

    # --- Smart yaw selection helpers ---
    def get_current_pose_xy_yaw(self) -> Tuple[float, float, float]:
        """Return (x, y, yaw) from current pose; fallback to start corner/yaw."""
        try:
            p = self.navigator.getCurrentPose()
            if p is not None:
                x = p.pose.position.x
                y = p.pose.position.y
                qx = p.pose.orientation.x
                qy = p.pose.orientation.y
                qz = p.pose.orientation.z
                qw = p.pose.orientation.w
                yaw = yaw_from_quat(qx, qy, qz, qw)
                return x, y, yaw
        except Exception:
            pass
        sx, sy = self.corners[self.start_corner]
        return sx, sy, self.start_yaw

    def choose_goal_yaw(self, gx: float, gy: float) -> float:
        """
        Smart policy:
          - If goal is close (<= close_goal_distance), keep current yaw to avoid double-rotate.
          - Else, face a look-ahead point (goal projected ahead by lookahead_distance).
        """
        if not self.use_goal_heading:
            return 0.0  # neutral; rely on tolerances

        cx, cy, cyaw = self.get_current_pose_xy_yaw()
        dx, dy = gx - cx, gy - cy
        dist = math.hypot(dx, dy)

        if dist <= max(self.close_goal_distance, 1e-3):
            return normalize_angle(cyaw)

        ux, uy = dx / dist, dy / dist
        ax, ay = gx + self.lookahead_distance * ux, gy + self.lookahead_distance * uy
        return normalize_angle(math.atan2(ay - cx, ax - cy))

    # --- Center utilities & transition to OOB active ---
    def center_of_quad(self) -> Tuple[float, float]:
        xs = [p[0] for p in self.quad_ccw]
        ys = [p[1] for p in self.quad_ccw]
        cx = sum(xs) / 4.0
        cy = sum(ys) / 4.0
        # Nudge to be at least safety_margin inside (handles skewed quads)
        cx, cy = push_inside_convex_quad(cx, cy, self.quad_ccw, margin=self.safety_margin)
        return cx, cy

    def go_to_center_and_enable_oob(self):
        cx, cy = self.center_of_quad()
        yaw = self.choose_goal_yaw(cx, cy)
        q = quaternion_from_euler(0.0, 0.0, yaw)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = cx
        goal.pose.position.y = cy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x = q[0]
        goal.pose.orientation.y = q[1]
        goal.pose.orientation.z = q[2]
        goal.pose.orientation.w = q[3]

        self.get_logger().info(f"Calibration complete. Moving to center ({cx:.2f},{cy:.2f}) before enabling OOB.")
        try:
            self.navigator.goToPose(goal)
            while not self.navigator.isTaskComplete():
                rclpy.spin_once(self, timeout_sec=0.5)
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info("Reached center. Enabling out-of-bounds logic.")
                self.oob_active = True
            else:
                self.get_logger().warn(f"Center move failed (result={result}). Enabling OOB anyway.")
                self.oob_active = True
        except Exception as e:
            self.get_logger().error(f"Exception going to center: {e}\n{traceback.format_exc()}")
            self.oob_active = True  # enable regardless, to be safe

    # --- OOB recovery check (called inside navigation loops) ---
    def maybe_recover_if_oob(self, last_check_time: List[float]) -> None:
        """
        If robot is outside the quad, send a corrective pose to push it back inside.
        last_check_time: a single-element list storing last check timestamp (mutable).
        """
        if not (self.oob_active and self.oob_recovery_enabled):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if last_check_time[0] is None or (now - last_check_time[0]) >= self.oob_check_period:
            last_check_time[0] = now
            cx, cy, _ = self.get_current_pose_xy_yaw()
            if point_outside_convex_quad(cx, cy, self.quad_ccw, tol=0.02):
                fix_x, fix_y = push_inside_convex_quad(cx, cy, self.quad_ccw, margin=self.safety_margin)
                q = quaternion_from_euler(0.0, 0.0, 0.0)
                fix = PoseStamped()
                fix.header.frame_id = 'map'
                fix.header.stamp = self.navigator.get_clock().now().to_msg()
                fix.pose.position.x = fix_x
                fix.pose.position.y = fix_y
                fix.pose.position.z = 0.0
                fix.pose.orientation.x, fix.pose.orientation.y, fix.pose.orientation.z, fix.pose.orientation.w = q
                self.get_logger().warn(
                    f"Out of bounds at ({cx:.2f},{cy:.2f}) — corrective pose to ({fix_x:.2f},{fix_y:.2f})."
                )
                try:
                    self.navigator.goToPose(fix)
                    while not self.navigator.isTaskComplete():
                        rclpy.spin_once(self, timeout_sec=0.5)
                    _ = self.navigator.getResult()
                except Exception as e:
                    self.get_logger().error(f"OOB recovery exception: {e}\n{traceback.format_exc()}")

    def calibration_lap(self):
        self.get_logger().info('Starting calibration lap...')
        order = ['top_left', 'top_right', 'bottom_right', 'bottom_left', 'top_left']
        for i in range(len(order) - 1):
            start = order[i]
            goal = order[i + 1]
            sx, sy = self.corners[start]
            gx, gy = self.corners[goal]

            # Nudge the corner goal inward by safety_margin to avoid edge fighting
            gx_n, gy_n = push_inside_convex_quad(gx, gy, self.quad_ccw, margin=self.safety_margin)

            if self.use_goal_heading:
                yaw = normalize_angle(yaw_towards(sx, sy, gx_n, gy_n))
            else:
                yaw = 0.0

            pose = self.corner_to_pose(goal, yaw)
            pose.pose.position.x = gx_n
            pose.pose.position.y = gy_n

            self.get_logger().info(
                f"Navigating from {start} ({sx:.2f},{sy:.2f}) to {goal}*nudged ({gx_n:.2f},{gy_n:.2f}) with yaw={yaw:.2f}"
            )
            try:
                self.navigator.goToPose(pose)
                # NO OOB checks during calibration
                while not self.navigator.isTaskComplete():
                    rclpy.spin_once(self, timeout_sec=1.0)
                result = self.navigator.getResult()
                if result == TaskResult.SUCCEEDED:
                    self.get_logger().info(f"Reached {goal} successfully.")
                else:
                    self.get_logger().warn(f"Failed to reach {goal}. Result: {result}")
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
                # Pick triangle and sample a point uniformly
                tri_idx = random.randint(0, 1)
                A, B, C = triangles[tri_idx]
                u, v = random.random(), random.random()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
                y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])

                # Hard-fence clamp: keep goal at least safety_margin inside
                x, y = push_inside_convex_quad(x, y, self.quad_ccw, margin=self.safety_margin)

                # Choose yaw using smart policy (uses current pose and clamped goal)
                yaw = self.choose_goal_yaw(x, y)

                # Build goal pose
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

                cx, cy, _ = self.get_current_pose_xy_yaw()
                self.get_logger().info(
                    f"Navigating to random goal {count+1}: "
                    f"from approx ({cx:.2f},{cy:.2f}) to ({x:.2f},{y:.2f}) with yaw={yaw:.2f}"
                )
                try:
                    self.navigator.goToPose(pose)
                    last_check_time = [None]
                    while not self.navigator.isTaskComplete():
                        rclpy.spin_once(self, timeout_sec=1.0)
                        # OOB checks are active now
                        self.maybe_recover_if_oob(last_check_time)
                    result = self.navigator.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info(f"Random goal {count+1} reached successfully.")
                    else:
                        self.get_logger().warn(f"Random goal {count+1} failed. Result: {result}")
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
