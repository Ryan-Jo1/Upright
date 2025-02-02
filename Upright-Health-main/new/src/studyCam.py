from flask import Flask, Blueprint, render_template, Response, send_from_directory
import cv2
import mediapipe as mp
import time
import threading
import queue
from openai import OpenAI
import pygame

study_app = Blueprint('study_app', __name__, static_folder="static", template_folder="templates")

# Initialize OpenAI API (replace with your API key or use environment variable)
client = OpenAI(api_key=("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"))

# Initialize pose estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Capture video from default camera
cap = None
camera_active = False
camera_lock = threading.Lock()

# Timer Setting
PREP_TIME = 60   # 5 minutes prep
STUDY_TIME = 1500 # 25 minutes study
countdown_time = PREP_TIME
timer_mode = "prep"
timer_running = False
timer_thread = None  # Timer thread variable

# --- Parameters for Posture Analysis ---
SHOULDER_LEVEL_DIFFERENCE_THRESHOLD = 28  # Strict threshold for uneven shoulders
HEAD_FORWARD_THRESHOLD = 35  # Strict threshold for head forward
AVERAGE_SHOULDER_HEIGHT_THRESHOLD = 0.75  # For slouching detection

# --- Time-Based Bad Feedback ---
BAD_POSTURE_TIME_THRESHOLD = 5  # Seconds of continuous bad posture before feedback
bad_posture_start_time = 0
bad_posture_detected = False

# --- XP Counter ---
GOOD_POSTURE_TIME_THRESHOLD = 5  # Seconds of continuous good posture for XP
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
FEEDBACK_INTERVAL = 5  # Generate feedback every 10 seconds

# Thread-safe access to genai feedback
genai_feedback_shared = ""  # Makes global GenAI feedback variable
genai_feedback_lock = threading.Lock()

# Queue for communication between threads
feedback_queue = queue.Queue()

# --- Pomodoro Timer ---
PREP_TIME = 60  # 5 minutes in seconds
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
# Load the MP3 file (replace with your file path)
pygame.mixer.music.load(r"C:/Users/hengj/OneDrive - stevens.edu/Upright-Health-main/Upright-Health-main/new/audio/badPostureSound.mp3")
current_volume = 0.1  # Initial volume
VOLUME_INCREASE_RATE = 0.05  # Rate of volume increase per second

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

    return response.choices[0].message.content.strip()

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

        genai_feedback = response.choices[0].message.content.strip()
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

def generate_frames():
    global camera_active
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


    while camera_active:  # Use the global flag here
        with camera_lock:
            if cap is None or not cap.isOpened():
                break

            ret, frame = cap.read()
            if not ret:
                break


        frame = cv2.flip(frame, 1)


        """# Display the Pomodoro Timer on the frame
        minutes = countdown_time // 60
        seconds = countdown_time % 60
        timer_text = f"{minutes:02d}:{seconds:02d}"

        cv2.putText(frame, timer_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
"""
        # Flip the image horizontally
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # --- Landmark Extraction ---
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * frame.shape[1],
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * frame.shape[0]]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * frame.shape[1],
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * frame.shape[0]]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE].x * frame.shape[1],
                    landmarks[mp_pose.PoseLandmark.NOSE].y * frame.shape[0]]

            if timer_mode == "study":
                # --- Calculations ---
                # 1. Shoulder Level Difference
                shoulder_level_diff = abs(left_shoulder[1] - right_shoulder[1])

                # 2. Head Forward Position
                shoulder_midpoint_x = (left_shoulder[0] + right_shoulder[0]) / 2
                head_forward_dist = nose[0] - shoulder_midpoint_x

                # 3. Average Shoulder Height
                average_shoulder_height = (left_shoulder[1] + right_shoulder[1]) / (2 * frame.shape[0])

                # --- Posture Analysis ---
                posture_feedback = "Good Posture!"
                feedback_color = (0, 255, 0)  # Green

                # 1. Shoulder Level Check (Unevenness)
                if shoulder_level_diff > SHOULDER_LEVEL_DIFFERENCE_THRESHOLD:
                    posture_feedback = "Shoulders uneven"
                    feedback_color = (0, 0, 255)  # Red

                # 2. Average Shoulder Height Check (Slouching)
                if average_shoulder_height > AVERAGE_SHOULDER_HEIGHT_THRESHOLD:
                    if posture_feedback == "Good Posture!":
                        posture_feedback = "Shoulders too low (Slouching)"
                        feedback_color = (0, 0, 255)
                    elif posture_feedback == "Shoulders uneven":
                        posture_feedback = "Shoulders uneven and too low"
                        feedback_color = (0, 0, 255)

                # 3. Head Forward Check
                if head_forward_dist > HEAD_FORWARD_THRESHOLD:
                    if posture_feedback == "Good Posture!":
                        posture_feedback = "Head forward"
                        feedback_color = (0, 0, 255)
                    elif posture_feedback == "Shoulders uneven":
                        posture_feedback = "Shoulders uneven, Head forward"
                        feedback_color = (0, 0, 255)
                    elif posture_feedback == "Shoulders too low (Slouching)":
                        posture_feedback = "Shoulders too low, Head forward"
                        feedback_color = (0, 0, 255)
                    elif posture_feedback == "Shoulders uneven and too low":
                        posture_feedback = "Shoulders uneven, too low, and Head forward"
                        feedback_color = (0, 0, 255)

                # --- Time-Based Feedback ---
                if posture_feedback != "Good Posture!":
                    if not bad_posture_detected:
                        bad_posture_start_time = time.time()
                        bad_posture_detected = True
                    else:
                        if time.time() - bad_posture_start_time > BAD_POSTURE_TIME_THRESHOLD:
                            posture_feedback += " - Please Correct Posture!"
                            feedback_color = (0, 0, 255)

                    # Reset good posture timer
                    good_posture_detected = False
                    xp_multiplier = 1  # Reset multiplier on bad posture
                    last_xp_increment_time = time.time()
                else:  # Good posture detected
                    bad_posture_detected = False  # Reset bad posture timer

                    if not good_posture_detected:
                        good_posture_start_time = time.time()
                        good_posture_detected = True
                        last_xp_increment_time = time.time()  # Reset XP increment time
                    else:
                        # Increment XP every second
                        if time.time() - last_xp_increment_time >= 1:
                            total_xp += xp_multiplier
                            last_xp_increment_time = time.time()

                        # --- Exponential XP Increase ---
                        if time.time() - xp_increase_timer >= XP_INCREASE_INTERVAL:
                            xp_multiplier = min(xp_multiplier * 2, 32)
                            xp_increase_timer = time.time()

                # --- GenAI Feedback ---
                if time.time() - last_feedback_time > FEEDBACK_INTERVAL:
                    feedback_queue.put(posture_feedback)
                    last_feedback_time = time.time()

                # Display posture feedback
                cv2.putText(frame, posture_feedback, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, feedback_color, 2)

                # Display XP counter and multiplier
                cv2.putText(frame, f"XP: {total_xp} (x{xp_multiplier})", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                # Get the latest genai feedback (thread-safe)
                with genai_feedback_lock:
                    local_genai_feedback = genai_feedback_shared

                # Calculate the maximum width for the wrapped text
                max_text_width = frame.shape[1] - 20  # Subtract 20 for padding (10 on each side)

                # Display genai feedback on the frame using the draw_wrapped_text function
                draw_wrapped_text(frame, local_genai_feedback, 10, 400, max_text_width, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # --- Audio Feedback ---
                if posture_feedback != "Good Posture!":
                    if not bad_posture_detected:
                        bad_posture_start_time = time.time()
                        bad_posture_detected = True

                    # Check if 5 seconds of bad posture have passed AND audio isn't already playing
                    if time.time() - bad_posture_start_time >= BAD_POSTURE_TIME_THRESHOLD and not audio_playing:
                        pygame.mixer.music.play(-1)
                        audio_playing = True
                        last_volume_increase_time = time.time()

                    # If audio is playing, increase volume every second
                    if audio_playing and time.time() - last_volume_increase_time >= 1:
                        current_volume = min(current_volume + VOLUME_INCREASE_RATE, 1.0)
                        pygame.mixer.music.set_volume(current_volume)
                        last_volume_increase_time = time.time()

                else:  # Good posture
                    if bad_posture_detected:  # Reset only when transitioning from bad to good posture
                        bad_posture_detected = False
                        bad_posture_start_time = 0
                    if audio_playing:
                        pygame.mixer.music.stop()
                        audio_playing = False
                        current_volume = 0.1

        # --- Display Pomodoro Timer ---
        if timer_running:
            minutes = countdown_time // 60
            seconds = countdown_time % 60
            timer_text = f"{minutes:02d}:{seconds:02d}"

            # Get text size for positioning
            (text_width, text_height), _ = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

            # Position the timer in the top-right corner
            timer_x = frame.shape[1] - text_width - 10  # 10 pixels padding from the right edge
            timer_y = text_height + 10  # 10 pixels padding from the top

            cv2.putText(frame, timer_text, (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Display timer mode text
            if timer_mode == "prep":
                if countdown_time > 10:
                    mode_text = "Prep your study area!"
                else:
                    mode_text = "Ready? Starting study time now!"
            else:
                mode_text = "Study Time!"  # You can customize this message

            (mode_text_width, mode_text_height), _ = cv2.getTextSize(mode_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            mode_text_x = frame.shape[1] - mode_text_width - 10
            mode_text_y = timer_y + text_height + 10  # Position below the timer

            cv2.putText(frame, mode_text, (mode_text_x, mode_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if timer_paused:
                timer_text += " (Paused)"

            cv2.putText(
                frame, timer_text, (timer_x, timer_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
            )

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera when the loop exits
    cap.release()


@study_app.route('/')
def index():
    return render_template('index.html')

@study_app.route('/study')
def study():
    return render_template('study.html')

@study_app.route('/patient')
def patient():
    return render_template('patient.html')

@study_app.route('/free')
def free():
    return render_template('free.html')

@study_app.route('/demo')
def demo():
    return render_template('demo.html')

@study_app.route('/video_feed')
def video_feed():
    """Start a new webcam feed session and restart timer."""
    release_camera()  # Ensure old session is closed
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@study_app.route('/stop_program')
def stop_program():
    """Stop the webcam and reset the timer."""
    release_camera()
    return "STOPPED"

@study_app.route('/start_music')
def start_music():
    pygame.mixer.music.play(-1)  # Play music in a loop
    return "MUSIC_STARTED"

@study_app.route('/stop_music')
def stop_music():
    pygame.mixer.music.stop()
    return "MUSIC_STOPPED"

@study_app.route('/study_video_feed')
def study_video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@study_app.route('/stop_study_program')
def stop_study_program():
    release_camera()
    return "STOPPED"

if __name__ == '__main__':
    app = Flask(__name__)
    app.register_blueprint(study_app)
    app.run(debug=True)