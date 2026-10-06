import cv2
import mediapipe as mp
import threading

from integrated_gaze_filter import GazeFilter

from integrated_eye_detection import (
    detect_face_and_eyes
)

from integrated_gaze import (
    extract_gaze_features
)

from integrated_cursor import (
    CursorController
)

from integrated_dweell import (
    DwellClick
)

from integrated_calibration import (
    GazeCalibration
)

from integrated_keyboard import (
    AccessibleVirtualKeyboard
)


class GazeMate:

    def __init__(self):

        # ==================================================
        # RUNNING STATE
        # ==================================================

        self.running = True

        # ==================================================
        # CALIBRATION
        # ==================================================

        self.calibration = (
            GazeCalibration()
        )

        # ==================================================
        # KALMAN FILTER
        # ==================================================

        self.gaze_filter = GazeFilter(
            process_noise=0.01,
            measurement_noise=0.08
        )

        # ==================================================
        # CURSOR
        # ==================================================

        self.cursor = CursorController(
            smoothing=0.35
        )

        # ==================================================
        # DWELL CLICK
        # ==================================================

        self.dwell = DwellClick(
            dwell_time=2.0
        )

        # ==================================================
        # KEYBOARD
        # ==================================================

        self.keyboard = None

    # ======================================================
    # EYE TRACKING
    # ======================================================

    def eye_tracking(self):

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():

            print(
                "ERROR: Could not open webcam."
            )

            self.running = False

            return

        mp_face_mesh = (
            mp.solutions.face_mesh
        )

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:

            while self.running:

                success, frame = (
                    camera.read()
                )

                if not success:
                    continue

                # --------------------------------------------------
                # MIRROR CAMERA
                # --------------------------------------------------

                frame = cv2.flip(
                    frame,
                    1
                )

                # --------------------------------------------------
                # FACE / EYE DETECTION
                # --------------------------------------------------

                (
                    frame,
                    face_detected,
                    left_eye_detected,
                    right_eye_detected,
                    iris_detected,
                    landmarks
                ) = detect_face_and_eyes(
                    frame,
                    face_mesh
                )

                # --------------------------------------------------
                # HYBRID GAZE FEATURES
                # --------------------------------------------------

                if (
                    face_detected
                    and
                    iris_detected
                    and
                    landmarks is not None
                ):

                    face_landmarks = (
                        landmarks[0].landmark
                    )

                    features = (
                        extract_gaze_features(
                            face_landmarks
                        )
                    )

                    if features is not None:

                        # ==========================================
                        # RIDGE REGRESSION
                        # ==========================================

                        screen_position = (
                            self.calibration
                            .map_features(
                                features
                            )
                        )

                        if screen_position is not None:

                            raw_x, raw_y = (
                                screen_position
                            )

                            # ======================================
                            # KALMAN FILTER
                            # ======================================

                            filtered_x, filtered_y = (
                                self.gaze_filter.update(
                                    raw_x,
                                    raw_y
                                )
                            )

                            # ======================================
                            # CURSOR
                            # ======================================

                            self.cursor.move_cursor(
                                filtered_x,
                                filtered_y
                            )

                            # ======================================
                            # DWELL CLICK
                            # ======================================

                            self.dwell.update(
                                filtered_x,
                                filtered_y
                            )

                            # ======================================
                            # DISPLAY STATUS
                            # ======================================

                            cv2.putText(
                                frame,
                                f"Gaze X: "
                                f"{filtered_x:.2f}",
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                frame,
                                f"Gaze Y: "
                                f"{filtered_y:.2f}",
                                (20, 65),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                            cv2.putText(
                                frame,
                                "Hybrid Gaze: ACTIVE",
                                (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 0),
                                2
                            )

                        else:

                            cv2.putText(
                                frame,
                                "Gaze mapping unavailable",
                                (20, 35),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2
                            )

                    else:

                        cv2.putText(
                            frame,
                            "Gaze features unavailable",
                            (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                else:

                    cv2.putText(
                        frame,
                        "Face / eyes not detected",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )

                    # Reset filter when tracking is lost
                    self.gaze_filter.reset()

                    self.dwell.reset()

                # --------------------------------------------------
                # CAMERA WINDOW
                # --------------------------------------------------

                cv2.imshow(
                    "GazeMate Eye Tracking",
                    frame
                )

                # --------------------------------------------------
                # EXIT
                # --------------------------------------------------

                key = (
                    cv2.waitKey(1)
                    &
                    0xFF
                )

                if key == ord("q"):

                    self.running = False

                    break

        camera.release()

        cv2.destroyAllWindows()

    # ======================================================
    # START APPLICATION
    # ======================================================

    def start(self):

        print()
        print(
            "========================================"
        )

        print(
            "       GazeMate Starting"
        )

        print(
            "========================================"
        )

        print()

        # ==================================================
        # CALIBRATION
        # ==================================================

        print(
            "Starting 9-point calibration..."
        )

        calibration_success = (
            self.calibration.calibrate()
        )

        if not calibration_success:

            print()
            print(
                "Calibration failed."
            )

            print(
                "GazeMate will not start."
            )

            return

        print()
        print(
            "Calibration completed successfully."
        )

        # ==================================================
        # KEYBOARD
        # ==================================================

        print(
            "Opening virtual keyboard..."
        )

        self.keyboard = (
            AccessibleVirtualKeyboard()
        )

        # ==================================================
        # TRACKING THREAD
        # ==================================================

        tracking_thread = threading.Thread(
            target=self.eye_tracking,
            daemon=True
        )

        tracking_thread.start()

        # ==================================================
        # KEYBOARD LOOP
        # ==================================================

        self.keyboard.mainloop()

        # ==================================================
        # STOP
        # ==================================================

        self.running = False

        print(
            "GazeMate stopped."
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    app = GazeMate()

    app.start()