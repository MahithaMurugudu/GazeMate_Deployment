import numpy as np


# ==========================================================
# IRIS LANDMARKS
# ==========================================================

LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ==========================================================
# EYE CORNER LANDMARKS
# ==========================================================

LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33


# ==========================================================
# EYE TOP / BOTTOM LANDMARKS
# ==========================================================

LEFT_EYE_TOP = [386, 387, 388]
LEFT_EYE_BOTTOM = [374, 373, 390]

RIGHT_EYE_TOP = [159, 158, 157]
RIGHT_EYE_BOTTOM = [145, 144, 163]


# ==========================================================
# HEAD POSE LANDMARKS
# ==========================================================

NOSE = 1
CHIN = 152

LEFT_FACE_SIDE = 234
RIGHT_FACE_SIDE = 454

LEFT_EYE_POSE = 263
RIGHT_EYE_POSE = 33


# ==========================================================
# AVERAGE LANDMARK POSITION
# ==========================================================

def average_landmark(landmarks, indices):

    x_values = []
    y_values = []
    z_values = []

    for index in indices:

        point = landmarks[index]

        x_values.append(point.x)
        y_values.append(point.y)
        z_values.append(point.z)

    return (
        np.mean(x_values),
        np.mean(y_values),
        np.mean(z_values)
    )


# ==========================================================
# IRIS CENTER
# ==========================================================

def get_iris_center(landmarks, indices):

    x_values = []
    y_values = []
    z_values = []

    for index in indices:

        point = landmarks[index]

        x_values.append(point.x)
        y_values.append(point.y)
        z_values.append(point.z)

    return (
        np.mean(x_values),
        np.mean(y_values),
        np.mean(z_values)
    )


# ==========================================================
# ONE EYE FEATURES
# ==========================================================

def get_eye_features(
    landmarks,
    iris_indices,
    inner_index,
    outer_index,
    top_indices,
    bottom_indices
):

    iris_x, iris_y, iris_z = get_iris_center(
        landmarks,
        iris_indices
    )

    inner_x = landmarks[inner_index].x
    inner_y = landmarks[inner_index].y

    outer_x = landmarks[outer_index].x
    outer_y = landmarks[outer_index].y

    # ------------------------------------------------------
    # Eye width
    # ------------------------------------------------------

    eye_width = abs(
        outer_x - inner_x
    )

    if eye_width < 0.001:
        return None

    # ------------------------------------------------------
    # Horizontal iris position
    # ------------------------------------------------------

    eye_left = min(
        inner_x,
        outer_x
    )

    horizontal_ratio = (
        iris_x - eye_left
    ) / eye_width

    # ------------------------------------------------------
    # Eye vertical position
    # ------------------------------------------------------

    top_x, top_y, top_z = average_landmark(
        landmarks,
        top_indices
    )

    bottom_x, bottom_y, bottom_z = average_landmark(
        landmarks,
        bottom_indices
    )

    eye_height = abs(
        bottom_y - top_y
    )

    if eye_height < 0.001:
        return None

    eye_top = min(
        top_y,
        bottom_y
    )

    vertical_ratio = (
        iris_y - eye_top
    ) / eye_height

    return (
        horizontal_ratio,
        vertical_ratio
    )


# ==========================================================
# HEAD POSE
# ==========================================================

def calculate_head_pose(landmarks):

    nose = landmarks[NOSE]
    chin = landmarks[CHIN]

    left_side = landmarks[LEFT_FACE_SIDE]
    right_side = landmarks[RIGHT_FACE_SIDE]

    # ------------------------------------------------------
    # Face center
    # ------------------------------------------------------

    face_center_x = (
        left_side.x +
        right_side.x
    ) / 2.0

    face_center_y = (
        left_side.y +
        right_side.y
    ) / 2.0

    # ------------------------------------------------------
    # Face width
    # ------------------------------------------------------

    face_width = abs(
        right_side.x -
        left_side.x
    )

    if face_width < 0.001:
        return None

    # ------------------------------------------------------
    # YAW
    # ------------------------------------------------------

    yaw = (
        nose.x -
        face_center_x
    ) / face_width

    # ------------------------------------------------------
    # Face height
    # ------------------------------------------------------

    face_height = abs(
        chin.y -
        face_center_y
    )

    if face_height < 0.001:
        return None

    # ------------------------------------------------------
    # PITCH
    # ------------------------------------------------------

    pitch = (
        nose.y -
        face_center_y
    ) / face_height

    # ------------------------------------------------------
    # ROLL
    # ------------------------------------------------------

    left_eye = landmarks[
        LEFT_EYE_POSE
    ]

    right_eye = landmarks[
        RIGHT_EYE_POSE
    ]

    eye_dx = (
        right_eye.x -
        left_eye.x
    )

    eye_dy = (
        right_eye.y -
        left_eye.y
    )

    if abs(eye_dx) < 0.001:
        return None

    roll = (
        eye_dy /
        abs(eye_dx)
    )

    return (
        yaw,
        pitch,
        roll
    )


# ==========================================================
# MAIN FEATURE EXTRACTION
# ==========================================================

def extract_gaze_features(landmarks):

    if landmarks is None:
        return None

    # ------------------------------------------------------
    # LEFT EYE
    # ------------------------------------------------------

    left_eye = get_eye_features(
        landmarks,
        LEFT_IRIS,
        LEFT_EYE_INNER,
        LEFT_EYE_OUTER,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM
    )

    # ------------------------------------------------------
    # RIGHT EYE
    # ------------------------------------------------------

    right_eye = get_eye_features(
        landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_INNER,
        RIGHT_EYE_OUTER,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM
    )

    if (
        left_eye is None
        or right_eye is None
    ):
        return None

    left_x, left_y = left_eye
    right_x, right_y = right_eye

    # ------------------------------------------------------
    # HEAD POSE
    # ------------------------------------------------------

    head_pose = calculate_head_pose(
        landmarks
    )

    if head_pose is None:
        return None

    yaw, pitch, roll = head_pose

    # ------------------------------------------------------
    # FACE CENTER
    # ------------------------------------------------------

    face_center_x = (
        landmarks[
            LEFT_FACE_SIDE
        ].x
        +
        landmarks[
            RIGHT_FACE_SIDE
        ].x
    ) / 2.0

    face_center_y = (
        landmarks[
            LEFT_FACE_SIDE
        ].y
        +
        landmarks[
            RIGHT_FACE_SIDE
        ].y
    ) / 2.0

    # ------------------------------------------------------
    # FEATURE VECTOR
    # ------------------------------------------------------

    features = np.array(
        [
            left_x,
            left_y,

            right_x,
            right_y,

            yaw,
            pitch,
            roll,

            face_center_x,
            face_center_y
        ],
        dtype=np.float64
    )

    return features


# ==========================================================
# RAW GAZE FUNCTION
# ==========================================================
#
# Kept for compatibility with existing files.
# The NEW system does not use this for cursor mapping.
# ==========================================================

def estimate_gaze(
    landmarks,
    frame_width,
    frame_height
):

    features = extract_gaze_features(
        landmarks
    )

    if features is None:
        return None

    left_x = features[0]
    left_y = features[1]

    right_x = features[2]
    right_y = features[3]

    gaze_x = (
        left_x +
        right_x
    ) / 2.0

    gaze_y = (
        left_y +
        right_y
    ) / 2.0

    gaze_x = max(
        0.0,
        min(1.0, gaze_x)
    )

    gaze_y = max(
        0.0,
        min(1.0, gaze_y)
    )

    return (
        gaze_x,
        gaze_y
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "GazeMate - Hybrid Gaze Feature Extraction"
    )

    print()
    print(
        "Eye features:"
    )

    print(
        "- Left iris position"
    )

    print(
        "- Left vertical iris position"
    )

    print(
        "- Right iris position"
    )

    print(
        "- Right vertical iris position"
    )

    print()
    print(
        "Head features:"
    )

    print(
        "- Yaw"
    )

    print(
        "- Pitch"
    )

    print(
        "- Roll"
    )

    print()
    print(
        "Face position:"
    )

    print(
        "- Face X"
    )

    print(
        "- Face Y"
    )