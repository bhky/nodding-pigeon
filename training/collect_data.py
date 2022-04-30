"""
Collect landmark data from video files.
"""
from typing import Dict, List

import cv2
import numpy as np
from mediapipe.python.solutions import face_detection as mp_face


def video_to_landmarks(video_path: str) -> List[List[float]]:
    landmarks: List[List[float]] = []
    cap = cv2.VideoCapture(video_path)
    with mp_face.FaceDetection(model_selection=0,
                               min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                # End of video.
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

    cap.release()
    return landmarks


def main() -> None:
    video_path_dict = {
        "head_nodding": "",
        "head_turning": "",
        "undefined": "",
    }
    output_file_path = "landmarks.npz"

    landmark_dict: Dict[str, List[List[float]]] = {}
    for name, video_path in video_path_dict.items():
        landmark_dict[name] = video_to_landmarks(video_path)

    np.savez_compressed(output_file_path, **landmark_dict)


if __name__ == "__main__":
    main()
