#!/usr/bin/env python3
# coding=utf-8
"""
Sit → wait for thumbs-up → Stand

ROS 2 Humble (PuppyPi)
- Immediately plays sit.d6ac
- Watches the camera for a thumbs-up
- When detected, plays stand.d6ac and exits
"""

import sys
import os
import time
import subprocess
import threading
import argparse

import cv2
import mediapipe as mp

import rclpy
from rclpy.node import Node

# Add the puppypi_control path to access action group helpers
sys.path.append('/home/ubuntu/software/puppypi_control')
from action_group_control import runActionGroup  # waits when second arg is True


class SitStandOnThumbsUp(Node):
    def __init__(self, show_window=True, device_index=0):
        super().__init__('sit_stand_on_thumbs_up')

        self.show_window = show_window
        self.device_index = device_index

        # Try to ensure puppy_control is running
        self._puppy_proc = self._start_puppy_control_if_needed()

        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

    # ---------- puppy_control helpers ----------
    def _puppy_control_running(self) -> bool:
        try:
            out = subprocess.run(
                ['ros2', 'node', 'list'],
                capture_output=True, text=True, timeout=5
            )
            return 'puppy_control' in out.stdout
        except Exception:
            return False

    def _start_puppy_control_if_needed(self):
        if self._puppy_control_running():
            self.get_logger().info('puppy_control already running.')
            return None

        self.get_logger().info('Starting puppy_control...')
        try:
            proc = subprocess.Popen(
                ['ros2', 'run', 'puppy_control', 'puppy_control'],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Give it a couple seconds to spin up
            time.sleep(2.0)
            if self._puppy_control_running():
                self.get_logger().info('puppy_control started.')
                return proc
            else:
                self.get_logger().warn('Could not confirm puppy_control startup.')
                return proc
        except Exception as e:
            self.get_logger().error(f'Failed to start puppy_control: {e}')
            return None

    # ---------- Action group runners ----------
    def play_action(self, name: str):
        self.get_logger().info(f'Running action group: {name}')
        try:
            # wait=True so it blocks until completed
            runActionGroup(name, True)
        except Exception as e:
            self.get_logger().error(f'Action "{name}" failed: {e}')
            raise

    # ---------- Thumbs-up detection ----------
    @staticmethod
    def _thumbs_up_detected(hand_landmarks) -> bool:
        """
        Simple heuristic:
          - Thumb tip is ABOVE (smaller y) than its IP and MCP joints.
          - Other four fingertips are BELOW (greater y) than their PIP joints (folded).
        Coordinate note: in images, y increases downward, so "up" means smaller y.
        """
        lm = hand_landmarks.landmark

        # Indices (MediaPipe Hands):
        # Thumb:    1 CMC, 2 MCP, 3 IP, 4 TIP
        # Index:    5 MCP, 6 PIP, 7 DIP, 8 TIP
        # Middle:   9 MCP, 10 PIP, 11 DIP, 12 TIP
        # Ring:     13 MCP, 14 PIP, 15 DIP, 16 TIP
        # Pinky:    17 MCP, 18 PIP, 19 DIP, 20 TIP

        # Thumb up: tip.y < ip.y and tip.y < mcp.y
        thumb_up = (lm[4].y < lm[3].y) and (lm[4].y < lm[2].y)

        # Other fingers folded: tip.y > pip.y (since folded down)
        index_folded  = lm[8].y  > lm[6].y
        middle_folded = lm[12].y > lm[10].y
        ring_folded   = lm[16].y > lm[14].y
        pinky_folded  = lm[20].y > lm[18].y

        return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

    # ---------- Camera loop ----------
    def wait_for_thumbs_up(self):
        cap = cv2.VideoCapture(self.device_index)
        if not cap.isOpened():
            raise RuntimeError('Failed to open camera')

        self.get_logger().info('Show a thumbs-up to make the robot stand. Press q to quit.')

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warn('Camera frame grab failed.')
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                detected = False
                if res.multi_hand_landmarks:
                    for hand in res.multi_hand_landmarks:
                        if self._thumbs_up_detected(hand):
                            detected = True
                        # draw landmarks (optional)
                        if self.show_window:
                            self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

                if self.show_window:
                    if detected:
                        cv2.putText(frame, 'THUMBS UP DETECTED', (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow('Thumbs-Up Detector', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if detected:
                    return True
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()

        return False

    # ---------- Orchestration ----------
    def run(self):
        # Step 1: sit
        self.play_action('sit.d6ac')

        # Step 2: watch for thumbs-up
        got_thumbs_up = self.wait_for_thumbs_up()

        if got_thumbs_up:
            self.get_logger().info('Thumbs-up detected -> standing.')
            self.play_action('stand.d6ac')
        else:
            self.get_logger().info('No thumbs-up detected; exiting.')

        # Done. Clean up puppy_control if we started it.
        if self._puppy_proc is not None:
            try:
                self.get_logger().info('Stopping puppy_control we started...')
                self._puppy_proc.terminate()
                self._puppy_proc.wait(timeout=3)
            except Exception:
                try:
                    self._puppy_proc.kill()
                except Exception:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Sit → wait for thumbs-up → Stand')
    parser.add_argument('--no-window', action='store_true',
                        help='Run without an OpenCV preview window')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera index (default 0)')
    args = parser.parse_args()

    rclpy.init()

    node = SitStandOnThumbsUp(
        show_window=not args.no_window,
        device_index=args.camera
    )

    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user.')
    except Exception as e:
        node.get_logger().error(f'Unhandled error: {e}')
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
