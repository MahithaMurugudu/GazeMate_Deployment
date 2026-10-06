import numpy as np


class GazeFilter:

    def __init__(
        self,
        process_noise=0.01,
        measurement_noise=0.08
    ):

        # ==================================================
        # STATE
        #
        # [x, y, velocity_x, velocity_y]
        # ==================================================

        self.state = None

        # ==================================================
        # COVARIANCE
        # ==================================================

        self.P = np.eye(
            4,
            dtype=np.float64
        )

        # ==================================================
        # STATE TRANSITION
        # ==================================================

        self.A = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ],
            dtype=np.float64
        )

        # ==================================================
        # MEASUREMENT MATRIX
        # ==================================================

        self.H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0]
            ],
            dtype=np.float64
        )

        # ==================================================
        # PROCESS NOISE
        # ==================================================

        self.Q = (
            np.eye(
                4,
                dtype=np.float64
            )
            *
            process_noise
        )

        # ==================================================
        # MEASUREMENT NOISE
        # ==================================================

        self.R = (
            np.eye(
                2,
                dtype=np.float64
            )
            *
            measurement_noise
        )

    # ======================================================
    # UPDATE
    # ======================================================

    def update(
        self,
        gaze_x,
        gaze_y
    ):

        measurement = np.array(
            [
                gaze_x,
                gaze_y
            ],
            dtype=np.float64
        )

        # ==================================================
        # FIRST FRAME
        # ==================================================

        if self.state is None:

            self.state = np.array(
                [
                    gaze_x,
                    gaze_y,
                    0.0,
                    0.0
                ],
                dtype=np.float64
            )

            return (
                gaze_x,
                gaze_y
            )

        # ==================================================
        # PREDICTION
        # ==================================================

        predicted_state = (
            self.A @ self.state
        )

        predicted_P = (
            self.A
            @ self.P
            @ self.A.T
            +
            self.Q
        )

        # ==================================================
        # INNOVATION
        # ==================================================

        innovation = (
            measurement
            -
            self.H @ predicted_state
        )

        innovation_covariance = (
            self.H
            @ predicted_P
            @ self.H.T
            +
            self.R
        )

        # ==================================================
        # KALMAN GAIN
        # ==================================================

        kalman_gain = (
            predicted_P
            @ self.H.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )

        # ==================================================
        # UPDATE STATE
        # ==================================================

        self.state = (
            predicted_state
            +
            kalman_gain
            @ innovation
        )

        # ==================================================
        # UPDATE COVARIANCE
        # ==================================================

        identity = np.eye(
            4,
            dtype=np.float64
        )

        self.P = (
            identity
            -
            kalman_gain @ self.H
        ) @ predicted_P

        # ==================================================
        # LIMIT OUTPUT
        # ==================================================

        filtered_x = max(
            0.0,
            min(
                1.0,
                self.state[0]
            )
        )

        filtered_y = max(
            0.0,
            min(
                1.0,
                self.state[1]
            )
        )

        return (
            filtered_x,
            filtered_y
        )

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.state = None

        self.P = np.eye(
            4,
            dtype=np.float64
        )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print(
        "GazeMate - Kalman Gaze Filter"
    )

    filter_test = GazeFilter()

    test_values = [
        (0.50, 0.50),
        (0.51, 0.50),
        (0.49, 0.51),
        (0.52, 0.49),
        (0.50, 0.50)
    ]

    for x, y in test_values:

        result = filter_test.update(
            x,
            y
        )

        print(
            "Input:",
            round(x, 3),
            round(y, 3),
            " -> Output:",
            round(result[0], 3),
            round(result[1], 3)
        )