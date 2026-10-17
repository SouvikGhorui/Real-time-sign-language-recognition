import cv2
import time
import mediapipe as mp
import math
import numpy as np

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE
)

def extract_frame_features(pose_result, hand_result):
    """
    Returns a single 1D numpy array of consistent legth (218)
    """
    if pose_result.pose_landmarks:
        pose_lms = pose_result.pose_landmarks[0][:23]
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lms]).flatten()
    else:
        pose = np.zeros(23 * 4)

    lh = np.zeros(21 *3)
    rh = np.zeros(21 * 3)

    if hand_result.hand_landmarks:
        for idx , handedness in enumerate( hand_result.handedness):
            label = handedness[0].category_name
            lms = hand_result.hand_landmarks[idx]
            flat_lms = np.array([[lm.x, lm.y, lm.z] for lm in lms]).flatten()

            if label == 'Left':
                lh = flat_lms
            elif label =='Right':
                rh = flat_lms
    return np.concatenate([pose, lh, rh])




# Create a hand landmarker instance with the image running mode
hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)

with HandLandmarker.create_from_options(hand_options) as hand_landmarker, PoseLandmarker.create_from_options(pose_options) as pose_landmarker:
    cap = cv2.VideoCapture(0)

    prev_time = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab the frame")
            break
        frame = cv2.flip(frame, 1)
        current_time = time.perf_counter()

        fps = 1 /(current_time - prev_time)
        prev_time = current_time

        cv2.putText(frame, f"Fps: {int(fps)}", (10,40),cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

        h, w , c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = rgb_frame)

        hand_result = hand_landmarker.detect(mp_image)
        pose_result = pose_landmarker.detect(mp_image)
        # detection_result = landmarker.detect(mp_image)

        features = extract_frame_features(pose_result, hand_result)
        cv2.putText(frame, f"Features Shape : {features.shape}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255,0), 2)

        h, w, c = frame.shape
        if hand_result.hand_landmarks:
            # We grab the first detected hand for the pinch challenge
            first_hand = hand_result.hand_landmarks[0]
            
            thumb_tip = first_hand[4]
            index_tip = first_hand[8]

            x4, y4 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            x8, y8 = int(index_tip.x * w), int(index_tip.y * h)

            distance = math.sqrt((x8 - x4)**2 + (y8 - y4)**2)
            cv2.line(frame, (x4, y4), (x8, y8), (255, 0, 0), 3)

            if distance < 30:
                cv2.putText(frame, "PINCH DETECTED!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Sign Language Challenges (Tasks API)', frame)

        # Clean exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()