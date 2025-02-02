import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow warnings

from flask import Flask, render_template, Response, request, jsonify, Blueprint
import cv2
import mediapipe as mp
import time
import threading
import queue
from openai import OpenAI
import pygame
import math

# Increase the recursion limit
sys.setrecursionlimit(3000)

app = Flask(__name__, static_folder="static", template_folder="templates")

patient_app = Blueprint('patient_app', __name__)

# Initialize OpenAI API (replace with your actual API key)
client = OpenAI(api_key="YOUR_ACTUAL_API_KEY_HERE")

# Initialize pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Capture video from default camera
cap = None
camera_active = False
camera_lock = threading.Lock()

# Timer Setting
PREP_TIME = 10   # 5 minutes prep
STUDY_TIME = 1500  # 25 minutes study
countdown_time = PREP_TIME
timer_mode = "prep"
timer_running = False
timer_thread = None  # Timer thread variable

# --- Parameters for Posture Analysis ---
SHOULDER_LEVEL_DIFFERENCE_THRESHOLD = 28  # Strict threshold for uneven shoulders
HEAD_FORWARD_THRESHOLD = 35  # Strict threshold for head forward
AVERAGE_SHOULDER_HEIGHT_THRESHOLD = 0.75  # For slouching detection

# --- Time-Based Bad Feedback ---
BAD_POSTURE_TIME_THRESHOLD = 10  # 10 Seconds of continuous bad posture before feedback
bad_posture_start_time = 0
bad_posture_detected = False

# --- XP Counter ---
GOOD_POSTURE_TIME_THRESHOLD = 10  # 10 Seconds of continuous good posture for XP
good_posture_start_time = 0
good_posture_detected = False
total_xp = 0
xp_multiplier = 1  # Initial XP multiplier
last_xp_increment_time = time.time()  # Track time of last XP increment

# --- Exponential XP counter ---
XP_INCREASE_INTERVAL = 10  # Seconds to increase XP multiplier
xp_increase_timer = time.time()

# --- GenAI Feedback hub ---
last_feedback_time = 0
FEEDBACK_INTERVAL = 15  # Generate feedback every 15 seconds

# Thread-safe access to genai feedback
genai_feedback_shared = ""  # Makes global GenAI feedback variable
genai_feedback_lock = threading.Lock()

# Queue for communication between threads
feedback_queue = queue.Queue()

# --- Pomodoro Timer ---
PREP_TIME = 15  # 5 minutes in seconds
STUDY_TIME = 1500  # 25 minutes in seconds
countdown_time = PREP_TIME
timer_mode = "prep"  # Initial mode
timer_running = True  # Flag to control the timer
timer_paused = False  # Flag to indicate if the timer is paused
pause_cooldown = False  # Flag to indicate if the timer is in cooldown
pause_cooldown_duration = 2  # Cooldown duration in seconds (DEFINE HERE)
pause_cooldown_start_time = 0  # Start time of cooldown
gesture_start_time = 0  # Time when the "OK" gesture is first detected
gesture_hold_duration = 2  # Seconds the gesture needs to be held
gesture_detected = False  # Flag to indicate if the "OK" gesture is currently detected

# --- Initialize pygame for audio ---
pygame.mixer.init()
# Load the MP3 file (replace with your actual file path)
pygame.mixer.music.load(r"C:/Users/hengj/OneDrive - stevens.edu/Upright-Health-main/Upright-Health-main/new/audio/badPostureSound.mp3")
current_volume = 0.1  # Initial volume
VOLUME_INCREASE_RATE = 0.05  # Rate of volume increase per second

# --- Parameters for Exercise Analysis ---
# Thresholds for two hand stretch extended upward
UPWARD_STRETCH_SHOULDER_ANGLE_THRESHOLD = 160  # Angle between shoulder, elbow, and wrist
UPWARD_STRETCH_HAND_HEIGHT_THRESHOLD = 0.9  # Hand height relative to frame height

# Thresholds for rotating arms in a circle
ARM_CIRCLE_SHOULDER_ANGLE_RANGE = (70, 110)  # Range of angles for shoulder abduction
ARM_CIRCLE_ELBOW_ANGLE_RANGE = (160, 180)  # Range of angles for elbow extension

# Thresholds for rotating circular stretch
CIRCULAR_STRETCH_SHOULDER_ANGLE_RANGE = (160, 180)  # Range of angles for shoulder flexion
CIRCULAR_STRETCH_ELBOW_ANGLE_RANGE = (160, 180)  # Range of angles for elbow extension

# Store posture data
good_posture_count = 0
bad_posture_count = 0

def initialize_camera():
    """Initialize the webcam for a new session."""
    global cap, camera_active
    with camera_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            camera_active = True
            initialize_timer()  # Restart timer when camera starts

def release_camera():
    """Release the webcam when stopping."""
    global cap, camera_active
    with camera_lock:
        if cap is not None:
            cap.release()
            cap = None
            camera_active = False
        stop_timer()  # Stop and reset timer when webcam stops

def initialize_timer():
    """Reset and start the Pomodoro timer."""
    global countdown_time, timer_mode, timer_running, timer_thread
    countdown_time = PREP_TIME  # Reset timer
    timer_mode = "prep"
    timer_running = True

    if timer_thread is None or not timer_thread.is_alive():
        timer_thread = threading.Thread(target=update_timer)
        timer_thread.start()

def stop_timer():
    """Stop the Pomodoro timer."""
    global timer_running
    timer_running = False

def update_timer():
    """Runs the Pomodoro timer logic."""
    global countdown_time, timer_mode, timer_running

    while timer_running:
        time.sleep(1)
        if countdown_time > 0:
            countdown_time -= 1
        else:
            # Switch between prep and study mode
            if timer_mode == "prep":
                timer_mode = "study"
                countdown_time = STUDY_TIME
            else:
                timer_mode = "prep"
                countdown_time = PREP_TIME

# Start the timer thread
timer_thread = threading.Thread(target=update_timer)
timer_thread.start()

# Function to generate personalized feedback using OpenAI
def generate_genai_feedback(posture_feedback):
    prompt = f"""
    The user's posture is currently classified as: {posture_feedback}.
    Based on this, provide a personalized, actionable recommendation to in under 10 words.
    """

    # Use the new OpenAI API syntax
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # Use the GPT-3.5 Turbo model
        messages=[
            {"role": "system", "content": "You are a helpful posture coach."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=20,
        temperature=0.7,
    )

    return response.choices.message.content.strip()

# Thread function for OpenAI API calls
def genai_feedback_thread():
    global genai_feedback_shared
    while True:
        posture_feedback = feedback_queue.get()
        if posture_feedback is None:
            break

        # Access the relevant time variables
        good_time = time.time() - good_posture_start_time if good_posture_detected else 0
        bad_time = time.time() - bad_posture_start_time if bad_posture_detected else 0

        if posture_feedback == "Good Posture!":
            prompt = f"""
            The user has maintained good posture for {int(good_time)} seconds.
            They currently have {total_xp} XP with a x{xp_multiplier} multiplier.
            Provide positive reinforcement and encourage them to continue in 15 words or less.
            """
        else:
            prompt = f"""
            The user's current posture is classified as: {posture_feedback}.
            They have been in this posture for {int(bad_time)} seconds.
            Provide a short, actionable recommendation to improve their posture in 15 words or less.
            """

        # Generate feedback using OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful posture coach."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=70,  # Adjust as needed
            temperature=0.7,
        )

        genai_feedback = response.choices.message.content.strip()
        # Update the shared variable with the lock
        with genai_feedback_lock:
            genai_feedback_shared = genai_feedback
        print("Live Feedback:", genai_feedback)

# Start the OpenAI feedback thread
feedback_thread = threading.Thread(target=genai_feedback_thread)
feedback_thread.start()

# --- Main Loop ---
def draw_wrapped_text(image, text, x, y, max_width, font, font_scale, color, thickness):
    """Draws wrapped text on an image."""
    lines = []
    current_line = ""

    # Get the font height immediately
    (text_width, text_height), _ = cv2.getTextSize("Test", font, font_scale, thickness)

    for word in text.split():
        test_line = current_line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

        if text_width <= max_width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word + " "
    lines.append(current_line)  # Add the last line

    for i, line in enumerate(lines):
        text_y = y + i * (text_height + 5)  # 5 is line spacing
        cv2.putText(image, line, (x, text_y), font, font_scale, color, thickness)

# Initialize audio playing flag
audio_playing = False
last_volume_increase_time = 0
bad_posture_time = 0  # Initialize bad posture time

# Helper function to calculate angle between three points
def calculate_angle(a, b, c):
    # Calculate vectors ba and bc
    ba = (a - b, a - b)
    bc = (c - b, c - b)

    # Calculate cosine of the angle using dot product and magnitudes
    cosine_angle = (ba * bc + ba * bc) / (
        (ba ** 2 + ba ** 2) ** 0.5 * (bc ** 2 + bc ** 2) ** 0.5
    )
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def generate_frames():
    global camera_active, good_posture_count, bad_posture_count
    initialize_camera()

    # Initialize all variables used in the function
    bad_posture_detected = False
    bad_posture_start_time = 0
    good_posture_detected = False
    good_posture_start_time = 0
    total_xp = 0
    xp_multiplier = 1
    last_xp_increment_time = time.time()
    xp_increase_timer = time.time()
    last_feedback_time = 0
    audio_playing = False
    last_volume_increase_time = 0
    current_volume = 0.1

    while camera_active:
        with camera_lock:
            if cap is None or not cap.isOpened():
                print("Camera is not opened")
                break

            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]]

            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if timer_mode == "study":
                shoulder_level_diff = abs(left_shoulder[1] - right_shoulder[1])
                shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
                head_forward_dist = nose[0] - shoulder_midpoint_x
                average_shoulder_height = (left_shoulder[1] + right_shoulder[1]) / (2 * frame.shape[0])
                posture_feedback = "Good Posture!"
                feedback_color = (0, 255, 0)
                if shoulder_level_diff > SHOULDER_LEVEL_DIFFERENCE_THRESHOLD:
                    posture_feedback = "Shoulders uneven"
                    feedback_color = (0, 0, 255)
                if average_shoulder_height > AVERAGE_SHOULDER_HEIGHT_THRESHOLD:
                    posture_feedback = "Shoulders too low (Slouching)"
                    feedback_color = (0, 0, 255)
                if head_forward_dist > HEAD_FORWARD_THRESHOLD:
                    posture_feedback = "Head forward"
                    feedback_color = (0, 0, 255)
                if posture_feedback != "Good Posture!":
                    if not bad_posture_detected:
                        bad_posture_start_time = time.time()
                        bad_posture_detected = True
                    if time.time() - bad_posture_start_time >= BAD_POSTURE_TIME_THRESHOLD and not audio_playing:
                        pygame.mixer.music.play(-1)
                        audio_playing = True
                        last_volume_increase_time = time.time()
                    if audio_playing and time.time() - last_volume_increase_time >= 1:
                        current_volume = min(current_volume + VOLUME_INCREASE_RATE, 1.0)
                        pygame.mixer.music.set_volume(current_volume)
                        last_volume_increase_time = time.time()
                else:
                    bad_posture_detected = False
                    if audio_playing:
                        pygame.mixer.music.stop()
                        audio_playing = False
                        current_volume = 0.1

                if time.time() - last_feedback_time > FEEDBACK_INTERVAL:
                    feedback_queue.put(posture_feedback)
                    last_feedback_time = time.time()
                cv2.putText(frame, posture_feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)
                cv2.putText(frame, f"XP: {total_xp} (x{xp_multiplier})", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                with genai_feedback_lock:
                    local_genai_feedback = genai_feedback_shared
                max_text_width = frame.shape[1] - 20
                draw_wrapped_text(frame, local_genai_feedback, 10, 400, max_text_width, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Check exercise metrics and display appropriate message
                if posture_feedback == "Good Posture!":
                    exercise_feedback = "Nice!"
                    exercise_color = (0, 255, 0)
                    good_posture_count += 1
                else:
                    exercise_feedback = "Bruh"
                    exercise_color = (0, 0, 255)
                    bad_posture_count += 1
                cv2.putText(frame, exercise_feedback, (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, exercise_color, 2)

        if timer_running:
            minutes = countdown_time // 60
            seconds = countdown_time % 60
            timer_text = f"{minutes:02d}:{seconds:02d}"
            (text_width, text_height), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            timer_x = frame.shape[1] - text_width - 10
            timer_y = text_height + 10
            cv2.putText(frame, timer_text, (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            mode_text = "Prep your study area!" if timer_mode == "prep" else "Study Time!"
            (mode_text_width, mode_text_height), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            mode_text_x = frame.shape[1] - mode_text_width - 10
            mode_text_y = timer_y + text_height + 10
            cv2.putText(frame, mode_text, (mode_text_x, mode_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/get_ai_feedback', methods=['POST'])
def get_ai_feedback():
    data = request.get_json()
    symptoms = data.get('symptoms', '')
    total_count = good_posture_count + bad_posture_count
    bad_percentage = (bad_posture_count / total_count) * 100 if total_count > 0 else 0

    prompt = f"""
    The user reported the following symptoms: {symptoms}.
    During the session, they had {bad_percentage:.2f}% bad posture.
    Based on this, recommend exercises or stretches to improve their posture.
    """

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful posture coach."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )

    feedback = response.choices[0].message['content'].strip()
    return jsonify({'feedback': feedback})

@app.route('/')
def index():
    return render_template('../frontend/templates/index.html')

@app.route('/study')
def study():
    return render_template('study.html')

@app.route('/patient')
def patient():
    return render_template('patient.html')


@app.route('/free')
def free():
    return render_template('free.html')


@app.route('/demo')
def demo():
    return render_template('demo.html')


@app.route('/video_feed')
def video_feed():
    """Start a new webcam feed session and restart timer."""
    release_camera()  # Ensure old session is closed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_program')
def stop_program():
    """Stop the webcam and reset the timer."""
    release_camera()
    return "STOPPED"

@patient_app.route('/patient_video_feed')
def patient_video_feed():
    """Start a new webcam feed session and restart timer."""
    release_camera()  # Ensure old session is closed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@patient_app.route('/stop_patient_program')
def stop_patient_program():
    """Stop the webcam and reset the timer."""
    release_camera()
    return "STOPPED"

# Register blueprint
app.register_blueprint(patient_app, url_prefix='/patient')

if __name__ == '__main__':
    app.run(debug=True)