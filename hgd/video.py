"""
Video utilities.
"""
from typing import List, Optional

import cv2
from mediapipe.python.solutions import face_detection as mp_face
from mediapipe.python.solutions import drawing_utils as mp_drawing

from hgd.config import Config


def video_to_landmarks(
        video_path: Optional[str],
        max_num_frames: Optional[int] = Config.seq_length,
        padding: bool = True
) -> List[List[float]]:
    video_path = video_path if video_path else 0  # For 0, webcam will be used.
    font_scale = 0.6
    line_thickness = 1

    valid_frame_count = 0
    landmarks: List[List[float]] = []
    cap = cv2.VideoCapture(video_path)  # pylint: disable=no-member
    with mp_face.FaceDetection(model_selection=0,
                               min_detection_confidence=0.5) as face_detection:

        while cap.isOpened():
            ret, bgr_frame = cap.read()
            if not ret:
                if video_path == 0:
                    continue  # Ignore empty frame of webcam.
                break  # End of given video.
            if max_num_frames and valid_frame_count >= max_num_frames:
                break

            frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # pylint: disable=no-member
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

            face_box_rel = detection.location_data.relative_bounding_box
            face_box_rel = [
                max(0.0, face_box_rel.xmin),
                max(0.0, face_box_rel.ymin),
                face_box_rel.width,
                face_box_rel.height,
            ]
            xs = [
                right_eye_rel.x,
                left_eye_rel.x,
                nose_tip_rel.x,
                mouth_center_rel.x,
                right_ear_rel.x,
                left_ear_rel.x,
            ]
            ys = [
                right_eye_rel.y,
                left_eye_rel.y,
                nose_tip_rel.y,
                mouth_center_rel.y,
                right_ear_rel.y,
                left_ear_rel.y,
            ]

            landmarks.append([*face_box_rel, *xs, *ys])

            valid_frame_count += 1

            if video_path == 0:
                mp_drawing.draw_detection(frame, detection)
                flipped_frame = cv2.flip(frame, 1)  # pylint: disable=no-member

                text = f"{valid_frame_count} / {max_num_frames}"
                cv2.putText(  # pylint: disable=no-member
                    flipped_frame, text, (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,  # pylint: disable=no-member
                    font_scale, (0, 0, 255),
                    line_thickness, cv2.LINE_AA  # pylint: disable=no-member
                )
                cv2.imshow(  # pylint: disable=no-member
                    "Webcam",
                    cv2.cvtColor(flipped_frame, cv2.COLOR_RGB2BGR)  # pylint: disable=no-member
                )

                if cv2.waitKey(1) == ord("q"):
                    break

    if padding and max_num_frames and len(landmarks) < max_num_frames:
        zeros = [0.0] * 12
        landmarks = landmarks + [zeros] * (max_num_frames - len(landmarks))

    cap.release()
    cv2.destroyAllWindows()
    return landmarks


if __name__ == "__main__":
    video_to_landmarks(None, None)
