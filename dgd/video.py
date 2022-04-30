"""
Video utilities.
"""
from typing import List, Optional

import cv2
from mediapipe.python.solutions import face_detection as mp_face


def video_to_landmarks(
        video_path: str,
        max_num_frames: Optional[int] = None,
        padding: bool = True
) -> List[List[float]]:
    valid_frame_count = 0
    landmarks: List[List[float]] = []
    cap = cv2.VideoCapture(video_path)
    with mp_face.FaceDetection(model_selection=0,
                               min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                # End of video.
                break
            if max_num_frames and valid_frame_count >= max_num_frames:
                break

            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            result = face_detection.process(frame)

            if not result or not result.detections or len(result.detections) != 1:
                continue

            detection = result.detections[0]
            right_eye_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.RIGHT_EYE)
            left_eye_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EYE)
            nose_tip_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.NOSE_TIP)
            mouth_center_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.MOUTH_CENTER)
            right_ear_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.RIGHT_EAR_TRAGION)
            left_ear_rel = mp_face.get_key_point(detection, mp_face.FaceKeyPoint.LEFT_EAR_TRAGION)

            landmarks.append(
                [right_eye_rel.x, right_eye_rel.y,
                 left_eye_rel.x, left_eye_rel.y,
                 nose_tip_rel.x, nose_tip_rel.y,
                 mouth_center_rel.x, mouth_center_rel.y,
                 right_ear_rel.x, right_eye_rel.y,
                 left_ear_rel.x, left_eye_rel.y]
            )

            valid_frame_count += 1

    if padding and max_num_frames and len(landmarks) < max_num_frames:
        zeros = [0.0] * 12
        landmarks = landmarks + [zeros] * (max_num_frames - len(landmarks))

    cap.release()
    return landmarks
