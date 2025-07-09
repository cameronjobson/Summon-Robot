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
import cv2
import mediapipe as mp
import argparse

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

def run_boxing_on_hand_detect():
    print("Starting camera for hand detection...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.7,
                           min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)
    hand_detected = False
    boxing_triggered = False
    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        if results.multi_hand_landmarks:
            if not boxing_triggered:
                print("Hand detected! Triggering boxing motion...")
                run_boxing_script()
                boxing_triggered = True
        else:
            boxing_triggered = False  # Reset trigger if hand is not detected
        # Draw hand landmarks if present
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Boxing Robot Launcher")
    parser.add_argument('--hand-detect', action='store_true', help='Trigger boxing action on hand detection via camera')
    args = parser.parse_args()
    if args.hand_detect:
        run_boxing_on_hand_detect()
        return 0
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