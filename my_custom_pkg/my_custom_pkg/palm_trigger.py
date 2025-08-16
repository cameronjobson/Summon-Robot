#!/usr/bin/env python3
# coding=utf-8
"""
Open Palm → trigger sit_then_stand_on_thumbs_up.py

- Watches the camera for an open palm (palm facing the camera).
- When detected, runs sit_then_stand_on_thumbs_up.py (headless) and exits.
"""

import sys
import os
import time
import argparse
import subprocess

import cv2
import mediapipe as mp


class PalmDetector:
    """
    Simple palm-facing-camera heuristic using MediaPipe Hands.
    - Fingers (index..pinky) extended: tip.y < pip.y
    - Thumb reasonably extended from palm (tip farther from wrist than IP)
    - Palm toward camera: mean (tip.z - dip.z) across fingers < 0
    """
    def __init__(self, camera_index: int = 0, show_window: bool = False):
        self.camera_index = camera_index
        self.show_window = show_window

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.drawer = mp.solutions.drawing_utils

    @staticmethod
    def _is_palm_facing(lm) -> bool:
        # Indices: thumb (2 MCP, 3 IP, 4 TIP)
        #          index (6 PIP, 8 TIP), middle (10,12), ring (14,16), pinky (18,20)
        def extended(tip, pip):  # smaller y = higher on image
            return lm[tip].y < lm[pip].y

        idx_ext = extended(8, 6)
        mid_ext = extended(12, 10)
        rng_ext = extended(16, 14)
        pky_ext = extended(20, 18)
        fingers_extended = idx_ext and mid_ext and rng_ext and pky_ext

        wrist = lm[0]
        thumb_farther = (abs(lm[4].x - wrist.x) + abs(lm[4].y - wrist.y)) > \
                        (abs(lm[3].x - wrist.x) + abs(lm[3].y - wrist.y))

        depth_diffs = [
            lm[8].z - lm[7].z,
            lm[12].z - lm[11].z,
            lm[16].z - lm[15].z,
            lm[20].z - lm[19].z
        ]
        palm_toward = sum(depth_diffs) / len(depth_diffs) < 0.0

        return fingers_extended and thumb_farther and palm_toward

    def wait_for_palm(self, timeout_sec: float | None = None) -> bool:
        # Prefer V4L2 on Linux/RPi
        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            print("ERROR: Could not open camera.")
            return False

        start = time.time()
        try:
            while True:
                # Timeout check
                if timeout_sec is not None and (time.time() - start) >= timeout_sec:
                    return False

                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.02)
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.hands.process(rgb)

                detected = False
                if res.multi_hand_landmarks:
                    for hand in res.multi_hand_landmarks:
                        if self._is_palm_facing(hand.landmark):
                            detected = True
                        if self.show_window:
                            self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

                if self.show_window:
                    if detected:
                        cv2.putText(frame, 'OPEN PALM DETECTED', (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.imshow('Palm Trigger', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return False  # user quit

                if detected:
                    return True
        finally:
            cap.release()
            if self.show_window:
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Open-palm trigger for sit_then_stand_on_thumbs_up.py')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    parser.add_argument('--timeout', type=float, default=0.0,
                        help='Seconds to wait (0 = no timeout). Default 0.')
    parser.add_argument('--show', action='store_true', help='Show a preview window with landmarks')
    parser.add_argument('--script', type=str, default=None,
                        help='Path to sit_then_stand_on_thumbs_up.py (defaults to same directory)')
    args = parser.parse_args()

    script_path = args.script or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'sit_then_stand_on_thumbs_up.py')

    if not os.path.exists(script_path):
        print(f"ERROR: Could not find script at {script_path}")
        return 1

    # Optional: mitigate X11 MIT-SHM issues if showing a window over VNC
    if args.show:
        os.environ.setdefault('GDK_DISABLE_SHM', '1')
        os.environ.setdefault('QT_X11_NO_MITSHM', '1')

    detector = PalmDetector(camera_index=args.camera, show_window=args.show)
    wait_secs = None if args.timeout == 0 else args.timeout

    print("Waiting for an OPEN PALM... (press 'q' to cancel if using --show)")
    detected = False
    try:
        detected = detector.wait_for_palm(timeout_sec=wait_secs)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130

    if not detected:
        print("No palm detected (timeout or cancelled). Exiting.")
        return 0

    print("Palm detected → running sit_then_stand_on_thumbs_up.py ...")
    try:
        # Run headless; that script can manage puppy_control if needed
        completed = subprocess.run([sys.executable, script_path, '--no-window'], timeout=120)
        if completed.returncode != 0:
            print(f"Warning: sit_then_stand_on_thumbs_up.py returned {completed.returncode}")
    except subprocess.TimeoutExpired:
        print("Warning: sit_then_stand_on_thumbs_up.py timed out.")
    except Exception as e:
        print(f"Failed to run sit_then_stand_on_thumbs_up.py: {e}")

    print("Done.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
