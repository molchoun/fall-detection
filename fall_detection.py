import numpy as np
import torch


class FallDetecion():
    ANGLE_WEIGHTS = {
        "hip_left": 0.1,
        "hip_right": 0.1,
        "knee_left": 0.1,
        "knee_right": 0.1,
        "elbow_left": 0.1,
        "elbow_right": 0.1,
        "torso": 0.4,
        "head": 0.4
    }
    num_skeleton = 10
    threshold = 0.5

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __call__(self, skeleton_cache=None):
        """
            This __call__ function takes a cache of skeletons as input, with a shape of (M x 17 x 2), where M represents the number of skeletons.
            The value of M is constant and represents time. For example, if you have a 7 fps stream and M is equal to 7 (M = 7), it means that the cache length is 1 second.
            The number 17 represents the count of points in each skeleton (as shown in skeleton.png), and 2 represents the (x, y) coordinates.

            This function uses the cache to detect falls.

            The function will return:
                - bool: isFall (True or False)
                - float: fallScore
        """
        isFall = False
        fallScore = None
        if not skeleton_cache:
            print("No skeleton was provided")
            return

        skeleton_cache = self.fill_missing_points(skeleton_cache)
        angles_array = self.calculate_angles(skeleton_cache)

        # Check if the change in angles exceeds the threshold
        diff = np.sum(np.diff(angles_array / 8)) / self.num_skeleton
        if diff >= self.threshold:
            isFall = True
            fallScore = diff

        return isFall, fallScore

    def calculate_angle(self, p1, p2, p3=None):
        """
            Calculates the angle between two vectors defined by three points.

            Args:
                p1 (numpy.ndarray): First point (x, y).
                p2 (numpy.ndarray): Second point (x, y).
                p3 (numpy.ndarray, optional): Third point (x, y). Defaults to None in case of computing angle
                    between vector and vertical axis.

            Returns:
                float: Angle in degrees.

        """
        v1 = p2 - p1
        if p3 is None:
            v2 = np.array([0, 1])
        else:
            v2 = p3 - p1

        dot_product = np.dot(v1, v2)
        magnitudes = np.linalg.norm(v1) * np.linalg.norm(v2)
        if not magnitudes:
            return 0

        cosine_angle = dot_product / magnitudes

        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        if angle_deg > 90:
            angle_deg = abs(angle_deg - 180)

        return angle_deg

    def calculate_angles(self, skeletons):
        """
            Calculates weighted average angles for each skeleton in the cache.

            Args:
                skeletons (numpy.ndarray): Cache of skeletons with shape (M x 17 x 2).

            Returns:
                numpy.ndarray: Array of weighted average angles.

        """
        angles_array = np.array([])

        for skeleton in skeletons:
            if np.all(skeleton < 0):
                angles_array = np.append(angles_array, 0)
                continue
            head = skeleton[0]
            shoulder_l = skeleton[5]
            shoulder_r = skeleton[6]
            elbow_l = skeleton[7]
            elbow_r = skeleton[8]
            hip_l = skeleton[11]
            hip_r = skeleton[12]
            knee_l = skeleton[13]
            knee_r = skeleton[14]
            ankle_l = skeleton[15]
            ankle_r = skeleton[16]

            mid_shoulder = (shoulder_l + shoulder_r) / 2
            mid_hip = (hip_l + hip_r) / 2

            angle_head = self.calculate_angle(mid_hip, head)
            angle_arm_l = self.calculate_angle(shoulder_l, elbow_l, hip_l)
            angle_arm_r = self.calculate_angle(shoulder_r, elbow_r, hip_r)
            angle_hip_l = self.calculate_angle(hip_l, knee_l, hip_r)
            angle_hip_r = self.calculate_angle(hip_r, knee_r, hip_l)
            angle_torso = self.calculate_angle(mid_hip, mid_shoulder)
            angle_knee_l = self.calculate_angle(knee_l, ankle_l, hip_l)
            angle_knee_r = self.calculate_angle(knee_r, ankle_r, hip_r)

            weighted_average = (
                    self.ANGLE_WEIGHTS["head"] * angle_head +
                    self.ANGLE_WEIGHTS["elbow_left"] * angle_arm_l +
                    self.ANGLE_WEIGHTS["elbow_right"] * angle_arm_r +
                    self.ANGLE_WEIGHTS["hip_left"] * angle_hip_l +
                    self.ANGLE_WEIGHTS["hip_right"] * angle_hip_r +
                    self.ANGLE_WEIGHTS["torso"] * angle_torso +
                    self.ANGLE_WEIGHTS["knee_left"] * angle_knee_l +
                    self.ANGLE_WEIGHTS["knee_right"] * angle_knee_r
            )

            angles_array = np.append(angles_array, weighted_average)
        return angles_array

    def fill_missing_points(self, skeleton_cache):
        # Replace NaN values with -1
        skeleton_cache = np.where(np.isnan(skeleton_cache), -1, skeleton_cache)
        return skeleton_cache
