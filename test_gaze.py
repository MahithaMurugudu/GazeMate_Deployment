import numpy as np


NOSE = 1

LEFT_EYE = 33
RIGHT_EYE = 263

FOREHEAD = 10
CHIN = 152


def estimate_gaze(landmarks, frame_width, frame_height):

    if landmarks is None:
        return None

    nose = landmarks[NOSE]

    left_eye = landmarks[LEFT_EYE]
    right_eye = landmarks[RIGHT_EYE]

    forehead = landmarks[FOREHEAD]
    chin = landmarks[CHIN]

    # Find the center between both eyes
    eye_center_x = (
        left_eye.x + right_eye.x
    ) / 2

    eye_center_y = (
        left_eye.y + right_eye.y
    ) / 2

    # Face width reference
    eye_width = abs(
        right_eye.x - left_eye.x
    )

    # Face height reference
    face_height = abs(
        chin.y - forehead.y
    )

    if eye_width == 0 or face_height == 0:
        return None

    # Calculate facial position
    position_x = (
        (nose.x - eye_center_x)
        / eye_width
    )

    position_y = (
        (nose.y - eye_center_y)
        / face_height
    )

    # Convert position into normalized screen coordinates
    gaze_x = 0.5 + (position_x * 1.5)
    gaze_y = 0.5 + (position_y * 3.0)

    # Keep coordinates inside screen range
    gaze_x = max(
        0.0,
        min(1.0, gaze_x)
    )

    gaze_y = max(
        0.0,
        min(1.0, gaze_y)
    )

    return gaze_x, gaze_y


if __name__ == "__main__":

    print("GazeMate - Gaze Estimation")
    print("Directional control is active.")