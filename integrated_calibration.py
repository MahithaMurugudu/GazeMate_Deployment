import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import json
import os
import time

from integrated_gaze import extract_gaze_features


# ==========================================================
# SIMPLE RIDGE REGRESSION
# NumPy implementation
# ==========================================================

class SimpleRidge:

    def __init__(self, alpha=1.0):

        self.alpha = alpha

        self.coef_ = None
        self.intercept_ = 0.0

    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    def fit(self, X, y):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        y = np.asarray(
            y,
            dtype=np.float64
        )

        # Add bias column
        X_bias = np.column_stack(
            (
                np.ones(X.shape[0]),
                X
            )
        )

        # --------------------------------------------------
        # Ridge equation
        #
        # beta = (X'X + alpha I)^-1 X'y
        #
        # Do NOT regularize intercept
        # --------------------------------------------------

        identity = np.eye(
            X_bias.shape[1],
            dtype=np.float64
        )

        identity[0, 0] = 0.0

        matrix = (
            X_bias.T @ X_bias
            +
            self.alpha * identity
        )

        vector = (
            X_bias.T @ y
        )

        # Solve instead of explicitly calculating inverse
        beta = np.linalg.solve(
            matrix,
            vector
        )

        self.intercept_ = float(
            beta[0]
        )

        self.coef_ = beta[1:]

        return self

    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    def predict(self, X):

        X = np.asarray(
            X,
            dtype=np.float64
        )

        if X.ndim == 1:

            X = X.reshape(
                1,
                -1
            )

        return (
            X @ self.coef_
            +
            self.intercept_
        )


# ==========================================================
# GAZE CALIBRATION
# ==========================================================

class GazeCalibration:

    def __init__(self):

        # ==================================================
        # SCREEN SIZE
        # ==================================================

        self.screen_width, self.screen_height = (
            pyautogui.size()
        )

        # ==================================================
        # CALIBRATION FILE
        # ==================================================

        self.save_path = (
            "calibration_data/calibration_data.json"
        )

        # ==================================================
        # 9-POINT CALIBRATION
        # ==================================================

        self.points = [

            (0.15, 0.15),
            (0.50, 0.15),
            (0.85, 0.15),

            (0.15, 0.50),
            (0.50, 0.50),
            (0.85, 0.50),

            (0.15, 0.85),
            (0.50, 0.85),
            (0.85, 0.85)

        ]

        # ==================================================
        # RIDGE MODELS
        # ==================================================

        self.model_x = None
        self.model_y = None

    # ======================================================
    # CALIBRATION
    # ======================================================

    def calibrate(self):

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            print(
                "ERROR: Could not open webcam."
            )

            return False

        mp_face_mesh = (
            mp.solutions.face_mesh
        )

        window_name = (
            "GazeMate Calibration"
        )

        cv2.namedWindow(
            window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.setWindowProperty(
            window_name,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN
        )

        all_features = []
        all_screen_x = []
        all_screen_y = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            # ==================================================
            # CALIBRATION POINTS
            # ==================================================

            for point_number, point in enumerate(
                self.points
            ):

                target_x = int(
                    point[0] *
                    self.screen_width
                )

                target_y = int(
                    point[1] *
                    self.screen_height
                )

                # ==============================================
                # STABILIZATION
                # ==============================================

                stabilization_start = (
                    time.time()
                )

                while (
                    time.time()
                    -
                    stabilization_start
                    <
                    1.0
                ):

                    success, frame = (
                        camera.read()
                    )

                    if not success:
                        continue

                    frame = cv2.flip(
                        frame,
                        1
                    )

                    display = np.zeros(
                        (
                            self.screen_height,
                            self.screen_width,
                            3
                        ),
                        dtype=np.uint8
                    )

                    cv2.circle(
                        display,
                        (
                            target_x,
                            target_y
                        ),
                        18,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        display,
                        "Look at the yellow circle",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Calibration point "
                        f"{point_number + 1}/"
                        f"{len(self.points)}",
                        (50, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        "Stabilizing...",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 220, 255),
                        2
                    )

                    cv2.imshow(
                        window_name,
                        display
                    )

                    key = (
                        cv2.waitKey(1)
                        &
                        0xFF
                    )

                    if key == 27:

                        camera.release()
                        cv2.destroyAllWindows()

                        return False

                # ==============================================
                # SAMPLE COLLECTION
                # ==============================================

                collection_start = (
                    time.time()
                )

                point_samples = 0

                while (
                    time.time()
                    -
                    collection_start
                    <
                    2.0
                ):

                    success, frame = (
                        camera.read()
                    )

                    if not success:
                        continue

                    frame = cv2.flip(
                        frame,
                        1
                    )

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    results = (
                        face_mesh.process(
                            rgb
                        )
                    )

                    display = np.zeros(
                        (
                            self.screen_height,
                            self.screen_width,
                            3
                        ),
                        dtype=np.uint8
                    )

                    cv2.circle(
                        display,
                        (
                            target_x,
                            target_y
                        ),
                        18,
                        (0, 255, 255),
                        -1
                    )

                    cv2.putText(
                        display,
                        "Keep looking at the circle",
                        (50, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        f"Calibration point "
                        f"{point_number + 1}/"
                        f"{len(self.points)}",
                        (50, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2
                    )

                    if results.multi_face_landmarks:

                        landmarks = (
                            results
                            .multi_face_landmarks[0]
                            .landmark
                        )

                        features = (
                            extract_gaze_features(
                                landmarks
                            )
                        )

                        if features is not None:

                            all_features.append(
                                features
                            )

                            all_screen_x.append(
                                point[0]
                            )

                            all_screen_y.append(
                                point[1]
                            )

                            point_samples += 1

                    cv2.putText(
                        display,
                        f"Samples: "
                        f"{point_samples}",
                        (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (180, 220, 255),
                        2
                    )

                    cv2.putText(
                        display,
                        "Press ESC to cancel",
                        (50, 195),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (180, 180, 180),
                        2
                    )

                    cv2.imshow(
                        window_name,
                        display
                    )

                    key = (
                        cv2.waitKey(1)
                        &
                        0xFF
                    )

                    if key == 27:

                        camera.release()
                        cv2.destroyAllWindows()

                        return False

                print(
                    f"Point "
                    f"{point_number + 1}: "
                    f"{point_samples} samples"
                )

        camera.release()
        cv2.destroyAllWindows()

        # ==================================================
        # CHECK DATA
        # ==================================================

        if len(all_features) < 50:

            print()
            print(
                "Calibration failed."
            )

            print(
                "Not enough valid samples."
            )

            return False

        # ==================================================
        # CONVERT DATA
        # ==================================================

        X = np.array(
            all_features,
            dtype=np.float64
        )

        Y_x = np.array(
            all_screen_x,
            dtype=np.float64
        )

        Y_y = np.array(
            all_screen_y,
            dtype=np.float64
        )

        print()
        print(
            "Total calibration samples:",
            len(X)
        )

        print(
            "Feature count:",
            X.shape[1]
        )

        # ==================================================
        # RIDGE REGRESSION
        # ==================================================

        self.model_x = SimpleRidge(
            alpha=1.0
        )

        self.model_y = SimpleRidge(
            alpha=1.0
        )

        self.model_x.fit(
            X,
            Y_x
        )

        self.model_y.fit(
            X,
            Y_y
        )

        # ==================================================
        # TRAINING ERROR
        # ==================================================

        predicted_x = (
            self.model_x.predict(X)
        )

        predicted_y = (
            self.model_y.predict(X)
        )

        errors = np.sqrt(
            (
                predicted_x - Y_x
            ) ** 2
            +
            (
                predicted_y - Y_y
            ) ** 2
        )

        average_error = np.mean(
            errors
        )

        maximum_error = np.max(
            errors
        )

        print()
        print(
            "Calibration completed."
        )

        print(
            f"Average calibration "
            f"training error: "
            f"{average_error:.4f}"
        )

        print(
            f"Maximum calibration "
            f"training error: "
            f"{maximum_error:.4f}"
        )

        # ==================================================
        # SAVE
        # ==================================================

        os.makedirs(
            "calibration_data",
            exist_ok=True
        )

        data = {

            "model": "ridge_numpy",

            "alpha": 1.0,

            "feature_count": int(
                X.shape[1]
            ),

            "model_x": {

                "coef": (
                    self.model_x
                    .coef_
                    .tolist()
                ),

                "intercept": float(
                    self.model_x
                    .intercept_
                )

            },

            "model_y": {

                "coef": (
                    self.model_y
                    .coef_
                    .tolist()
                ),

                "intercept": float(
                    self.model_y
                    .intercept_
                )

            }

        }

        with open(
            self.save_path,
            "w"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )

        print(
            "Calibration data saved to:"
        )

        print(
            self.save_path
        )

        return True

    # ======================================================
    # LOAD CALIBRATION
    # ======================================================

    def load_calibration(self):

        if not os.path.exists(
            self.save_path
        ):

            print(
                "Calibration file does not exist."
            )

            return False

        try:

            with open(
                self.save_path,
                "r"
            ) as file:

                data = json.load(
                    file
                )

            if data.get(
                "model"
            ) != "ridge_numpy":

                print(
                    "Calibration file is not "
                    "a NumPy Ridge model."
                )

                return False

            self.model_x = SimpleRidge(
                alpha=data.get(
                    "alpha",
                    1.0
                )
            )

            self.model_y = SimpleRidge(
                alpha=data.get(
                    "alpha",
                    1.0
                )
            )

            self.model_x.coef_ = np.array(
                data["model_x"]["coef"],
                dtype=np.float64
            )

            self.model_y.coef_ = np.array(
                data["model_y"]["coef"],
                dtype=np.float64
            )

            self.model_x.intercept_ = (
                float(
                    data["model_x"]["intercept"]
                )
            )

            self.model_y.intercept_ = (
                float(
                    data["model_y"]["intercept"]
                )
            )

            print(
                "Ridge calibration loaded."
            )

            return True

        except Exception as error:

            print(
                "Could not load calibration:"
            )

            print(error)

            return False

    # ======================================================
    # MAP FEATURES
    # ======================================================

    def map_features(
        self,
        features
    ):

        if (
            self.model_x is None
            or self.model_y is None
        ):

            return None

        try:

            features = np.array(
                features,
                dtype=np.float64
            ).reshape(
                1,
                -1
            )

            screen_x = (
                self.model_x
                .predict(features)[0]
            )

            screen_y = (
                self.model_y
                .predict(features)[0]
            )

            screen_x = max(
                0.0,
                min(
                    1.0,
                    screen_x
                )
            )

            screen_y = max(
                0.0,
                min(
                    1.0,
                    screen_y
                )
            )

            return (
                screen_x,
                screen_y
            )

        except Exception as error:

            print(
                "Gaze mapping error:",
                error
            )

            return None


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "GazeMate - NumPy Ridge Calibration"
    )

    print(
        "Model: Ridge Regression"
    )

    print(
        "Implementation: NumPy"
    )

    print(
        "Calibration points: 9"
    )

    print(
        "Features: eye + head pose + face position"
    )