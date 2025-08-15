#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import math
import random
import sys
import traceback
from typing import List, Tuple
import os
import time
import subprocess

# --- Optional CV imports (only needed for palm check) ---
import cv2
import mediapipe as mp

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

# --- (Optional) soft geofence helpers (currently disabled by default) ---
def dot(ax: float, ay: float, bx: float, by: float) -> float:
    return ax * bx + ay * by

def norm(ax: float, ay: float) -> float:
    return math.hypot(ax, ay)

def push_inside_convex_quad(x: float, y: float, verts_ccw: List[Tuple[float, float]],
                            margin: float = 0.10, iters: int = 2) -> Tuple[float, float]:
    """
    Conservative inward push; ONLY use if explicitly enabled.
    """
    px, py = x, y
    for _ in range(iters):
        moved = False
        for i in range(4):
            x1, y1 = verts_ccw[i]
            x2, y2 = verts_ccw[(i + 1) % 4]
            ex, ey = (x2 - x1), (y2 - y1)
            # Use RIGHT normal (ey, -ex) as inward; flip if your quad winds the other way.
            nx, ny = (ey, -ex)
            nlen = norm(nx, ny)
            if nlen == 0.0:
                continue
            nx, ny = nx / nlen, ny / nlen
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
    for i in range(4):
        x1, y1 = verts_ccw[i]
        x2, y2 = verts_ccw[(i + 1) % 4]
        ex, ey = (x2 - x1), (y2 - y1)
        nx, ny = (ey, -ex)  # matches push_inside_convex_quad inward choice
        nlen = norm(nx, ny)
        if nlen == 0.0:
            continue
        nx, ny = nx / nlen, ny / nlen
        d = dot(x - x1, y - y1, nx, ny)
        if d < -tol:
            return True
    return False

# --- CV: palm (open hand) detector ---
class PalmDetector:
    """
    Simple palm-facing-camera heuristic using MediaPipe Hands.
    - Fingers (index..pinky) extended: tip.y < pip.y
    - Thumb reasonably extended from palm (tip farther from wrist than IP)
    - Palm "toward camera" approximation: average of finger DIP->TIP z deltas negative (tips closer).
      (MediaPipe z is roughly depth; more negative ~= closer to camera.)
    """
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

    def _is_palm_facing(self, lm) -> bool:
        # Indices
        # Thumb: 2 MCP, 3 IP, 4 TIP
        # Fingers: (index) 6 PIP, 8 TIP; (middle) 10,12; (ring) 14,16; (pinky) 18,20
        def extended(tip, pip):  # smaller y = higher in image
            return lm[tip].y < lm[pip].y

        idx_ext = extended(8, 6)
        mid_ext = extended(12, 10)
        rng_ext = extended(16, 14)
        pky_ext = extended(20, 18)
        fingers_extended = idx_ext and mid_ext and rng_ext and pky_ext

        # Thumb: TIP further from wrist (0) than IP to avoid fist
        wrist = lm[0]
        thumb_farther = (abs(lm[4].x - wrist.x) + abs(lm[4].y - wrist.y)) > (abs(lm[3].x - wrist.x) + abs(lm[3].y - wrist.y))

        # Palm facing camera: average (tip.z - dip.z) < 0 across extended fingers
        # Use PIP/DIP proxies since DIP not used above
        # DIP indices: index 7, middle 11, ring 15, pinky 19
        depth_diffs = [
            lm[8].z - lm[7].z,
            lm[12].z - lm[11].z,
            lm[16].z - lm[15].z,
            lm[20].z - lm[19].z
        ]
        palm_toward = sum(depth_diffs) / len(depth_diffs) < 0.0

        return fingers_extended and thumb_farther and palm_toward

    def wait_for_palm(self, timeout_sec: float = 20.0) -> bool:
        # Prefer V4L2 on RPi
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            return False

        start = time.time()
        try:
            while (time.time() - start) < timeout_sec:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)
                if res.multi_hand_landmarks:
                    for hand in res.multi_hand_landmarks:
                        if self._is_palm_facing(hand.landmark):
                            return True
        finally:
            cap.release()
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

        # OOB controls (active only after center)
        self.declare_parameter('oob_recovery_enabled', True)
        self.declare_parameter('oob_check_period', 1.0)        # seconds

        # (Optional) clamping toggles — default OFF to avoid altering your geometry
        self.declare_parameter('clamp_random_goals', False)
        self.declare_parameter('nudge_corners', False)
        self.declare_parameter('safety_margin', 0.10)          # used only if above toggles are True

        # CV parameters
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('palm_timeout_sec', 20.0)

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

        self.oob_recovery_enabled = self.get_parameter('oob_recovery_enabled').get_parameter_value().bool_value
        self.oob_check_period = self.get_parameter('oob_check_period').get_parameter_value().double_value

        self.clamp_random_goals = self.get_parameter('clamp_random_goals').get_parameter_value().bool_value
        self.nudge_corners = self.get_parameter('nudge_corners').get_parameter_value().bool_value
        self.safety_margin = self.get_parameter('safety_margin').get_parameter_value().double_value

        self.camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.palm_timeout_sec = self.get_parameter('palm_timeout_sec').get_parameter_value().double_value

        # Validate corners
        for name in ['top_left', 'top_right', 'bottom_right', 'bottom_left']:
            if len(self.corners[name]) != 2:
                self.get_logger().error(f"Corner {name} must be a list of two floats [x, y].")
                sys.exit(1)

        # Quad order (as you had it)
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
        self.get_logger().info('Waiting for Nav2 to become active...')
        self.navigator.waitUntilNav2Active()
        self.get_logger().info('Nav2 is active.')

        # Try to apply speed limits (if supported)
        try:
            self.navigator.setSpeedLimits(self.max_speed, self.max_accel, 0.0)
            self.get_logger().info(f"Speed limits set: max_speed={self.max_speed}, max_accel={self.max_accel}")
        except Exception as e:
            self.get_logger().warn(f"setSpeedLimits not supported or failed: {e}")

        # Initial pose
        self.set_initial_pose()
        rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f"Waiting {self.amcl_wait_time} seconds for AMCL convergence...")
        time.sleep(self.amcl_wait_time)

        # Calibration lap (no OOB, no corner nudging)
        self.calibration_lap()

        # Go to center; once reached, enable OOB logic
        self.go_to_center_and_enable_oob()

        # Prepare palm detector (used only for random goals)
        self.palm_detector = PalmDetector(camera_index=self.camera_index)

        # Random goal loop (OOB logic is now active)
        self.random_goal_loop()

    def set_initial_pose(self):
        pose = self.corner_to_pose(self.start_corner, self.start_yaw)
        pose.header.stamp = rclpy.time.Time().to_msg()  # time=0 (use latest TF)
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
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q
        return pose

    # --- Smart yaw selection helpers ---
    def get_current_pose_xy_yaw(self) -> Tuple[float, float, float]:
        try:
            p = self.navigator.getCurrentPose()
            if p is not None:
                x = p.pose.position.x
                y = p.pose.position.y
                qx, qy, qz, qw = (p.pose.orientation.x, p.pose.orientation.y,
                                  p.pose.orientation.z, p.pose.orientation.w)
                yaw = yaw_from_quat(qx, qy, qz, qw)
                return x, y, yaw
        except Exception:
            pass
        sx, sy = self.corners[self.start_corner]
        return sx, sy, self.start_yaw

    def choose_goal_yaw(self, gx: float, gy: float) -> float:
        if not self.use_goal_heading:
            return 0.0
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
        return (sum(xs) / 4.0, sum(ys) / 4.0)

    def go_to_center_and_enable_oob(self):
        cx, cy = self.center_of_quad()
        self.get_logger().info(f"Calibration complete. Moving to center ({cx:.2f},{cy:.2f}) before enabling OOB.")
        yaw = self.choose_goal_yaw(cx, cy)
        q = quaternion_from_euler(0.0, 0.0, yaw)
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.navigator.get_clock().now().to_msg()
        goal.pose.position.x = cx
        goal.pose.position.y = cy
        goal.pose.position.z = 0.0
        goal.pose.orientation.x, goal.pose.orientation.y, goal.pose.orientation.z, goal.pose.orientation.w = q

        try:
            self.navigator.goToPose(goal)
            while not self.navigator.isTaskComplete():
                rclpy.spin_once(self, timeout_sec=0.5)
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.get_logger().info("Reached center. Enabling out-of-bounds logic.")
            else:
                self.get_logger().warn(f"Center move failed (result={result}). Enabling OOB anyway.")
        except Exception as e:
            self.get_logger().error(f"Exception going to center: {e}\n{traceback.format_exc()}")
        self.oob_active = True

    # --- OOB recovery check (called inside navigation loops *after* center) ---
    def maybe_recover_if_oob(self, last_check_time: List[float]) -> None:
        if not (self.oob_active and self.oob_recovery_enabled):
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if last_check_time[0] is None or (now - last_check_time[0]) >= self.oob_check_period:
            last_check_time[0] = now
            cx, cy, _ = self.get_current_pose_xy_yaw()
            if point_outside_convex_quad(cx, cy, self.quad_ccw, tol=0.02):
                # Simple corrective pose: go straight back to centroid
                fix_x, fix_y = self.center_of_quad()
                q = quaternion_from_euler(0.0, 0.0, 0.0)
                fix = PoseStamped()
                fix.header.frame_id = 'map'
                fix.header.stamp = self.navigator.get_clock().now().to_msg()
                fix.pose.position.x = fix_x
                fix.pose.position.y = fix_y
                fix.pose.position.z = 0.0
                fix.pose.orientation.x, fix.pose.orientation.y, fix.pose.orientation.z, fix.pose.orientation.w = q
                self.get_logger().warn(
                    f"Out of bounds at ({cx:.2f},{cy:.2f}) — corrective pose to center ({fix_x:.2f},{fix_y:.2f})."
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

            # NO corner nudging here — use the exact corners
            yaw = normalize_angle(yaw_towards(sx, sy, gx, gy)) if self.use_goal_heading else 0.0

            pose = self.corner_to_pose(goal, yaw)
            self.get_logger().info(
                f"Navigating from {start} ({sx:.2f},{sy:.2f}) to {goal} ({gx:.2f},{gy:.2f}) with yaw={yaw:.2f}"
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

    # --- Run the sit→thumbs-up→stand script if palm detected (random goals only) ---
    def maybe_run_sit_stand_on_palm(self):
        self.get_logger().info(f"Scanning for open palm for up to {self.palm_timeout_sec:.0f}s...")
        found = False
        try:
            found = self.palm_detector.wait_for_palm(timeout_sec=self.palm_timeout_sec)
        except Exception as e:
            self.get_logger().warn(f"Palm detection error (continuing anyway): {e}")

        if found:
            self.get_logger().info("Palm detected. Running sit_then_stand_on_thumbs_up.py ...")
            try:
                script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                           'sit_then_stand_on_thumbs_up.py')
                # Prefer calling with the current Python to avoid exec perms/env issues
                subprocess.run([sys.executable, script_path, '--no-window'], timeout=90)
                self.get_logger().info("sit_then_stand_on_thumbs_up.py completed.")
            except subprocess.TimeoutExpired:
                self.get_logger().warn("sit_then_stand_on_thumbs_up.py timed out; continuing.")
            except Exception as e:
                self.get_logger().warn(f"Failed to run sit_then_stand_on_thumbs_up.py: {e}")
        else:
            self.get_logger().info("No palm detected; continuing navigation.")

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
                # Uniform sample inside quad via two triangles
                tri_idx = random.randint(0, 1)
                A, B, C = triangles[tri_idx]
                u, v = random.random(), random.random()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                x = A[0] + u * (B[0] - A[0]) + v * (C[0] - A[0])
                y = A[1] + u * (B[1] - A[1]) + v * (C[1] - A[1])

                # Optional clamp (OFF by default). Your sampler is already inside.
                if self.clamp_random_goals:
                    x, y = push_inside_convex_quad(x, y, self.quad_ccw, margin=self.safety_margin)

                # Choose yaw using smart policy
                yaw = self.choose_goal_yaw(x, y)

                # Build goal pose
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.navigator.get_clock().now().to_msg()
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = 0.0
                q = quaternion_from_euler(0.0, 0.0, yaw)
                pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = q

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
                        self.maybe_recover_if_oob(last_check_time)
                    result = self.navigator.getResult()
                    if result == TaskResult.SUCCEEDED:
                        self.get_logger().info(f"Random goal {count+1} reached successfully.")
                        # === ONLY here: after a random-goal success ===
                        self.maybe_run_sit_stand_on_palm()
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
