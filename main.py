import cv2
import mediapipe as mp
import numpy as np
import time

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

left = [263, 387, 385, 362, 380, 373]
right = [33, 160, 158, 133, 153, 144]

def euclidean_dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def eye_aspect_ratio(landmarks, eye_indices, frame_shape):
    h, w, _ = frame_shape
    eye = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in eye_indices])
    p2_minus_p6 = euclidean_dist(eye[1], eye[5])
    p3_minus_p5 = euclidean_dist(eye[2], eye[4])
    p1_minus_p4 = euclidean_dist(eye[0], eye[3])
    return (p2_minus_p6 + p3_minus_p5) / (2.0 * p1_minus_p4)

# Sleep tracking state
asleep = False
sleep_start_time = None
total_sleep_time = 0
threshold = 0.25

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_ear = eye_aspect_ratio(face_landmarks.landmark, left, frame.shape)
            right_ear = eye_aspect_ratio(face_landmarks.landmark, right, frame.shape)
            ear = (left_ear + right_ear) / 2.0

            if ear < threshold:  # eyes closed
                if not asleep and sleep_start_time is None:
                    # eyes just closed --> start timing
                    sleep_start_time = time.time()
                elif not asleep and sleep_start_time is not None:
                    # check if eyes stayed closed long enough
                    if time.time() - sleep_start_time >= 3:
                        asleep = True
            else:  # eyes open
                if asleep:  # woke up after confirmed sleep
                    asleep = False
                    total_sleep_time += time.time() - sleep_start_time
                sleep_start_time = None  # reset if not asleep

    # Show status
    status = "Asleep" if asleep else "Awake"
    live_sleep_time = (time.time() - sleep_start_time) if (asleep and sleep_start_time) else 0
    total_time_display = int(total_sleep_time + live_sleep_time)

    cv2.putText(frame, f"Status: {status}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total Sleep: {total_time_display}s", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sleep Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
