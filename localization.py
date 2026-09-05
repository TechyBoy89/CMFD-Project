import cv2
import numpy as np


def find_copy_move(image):

    # --------------------------------
    # Create SIFT
    # --------------------------------

    sift = cv2.SIFT_create()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    keypoints, descriptors = sift.detectAndCompute(
        gray,
        None
    )

    if descriptors is None or len(keypoints) < 3:

        print("Not enough SIFT features found.")

        return image, False

    print("SIFT Keypoints :", len(keypoints))

    # --------------------------------
    # Feature Matching
    # --------------------------------

    matcher = cv2.BFMatcher(
        cv2.NORM_L2
    )

    matches = matcher.knnMatch(
        descriptors,
        descriptors,
        k=3
    )

    good_matches = []

    # --------------------------------
    # Check Matches
    # --------------------------------

    for match_list in matches:

        if len(match_list) < 3:
            continue

        m = match_list[1]
        n = match_list[2]

        # Lowe ratio test

        if m.distance < 0.7 * n.distance:

            point1 = keypoints[m.queryIdx].pt
            point2 = keypoints[m.trainIdx].pt

            dx = point1[0] - point2[0]
            dy = point1[1] - point2[1]

            spatial_distance = np.sqrt(
                dx * dx + dy * dy
            )

            # Ignore nearby points

            if spatial_distance > 40:

                good_matches.append(m)

    print("Good Matches :", len(good_matches))

    # --------------------------------
    # Create Result Image
    # --------------------------------

    result = image.copy()

    suspicious_points = []

    # --------------------------------
    # Draw Good Matches
    # --------------------------------

    for match in good_matches:

        point1 = keypoints[
            match.queryIdx
        ].pt

        point2 = keypoints[
            match.trainIdx
        ].pt

        x1 = int(point1[0])
        y1 = int(point1[1])

        x2 = int(point2[0])
        y2 = int(point2[1])

        suspicious_points.append(
            (x1, y1)
        )

        suspicious_points.append(
            (x2, y2)
        )

        # First point

        cv2.circle(
            result,
            (x1, y1),
            4,
            (0, 0, 255),
            -1
        )

        # Second point

        cv2.circle(
            result,
            (x2, y2),
            4,
            (0, 0, 255),
            -1
        )

        # Matching line

        cv2.line(
            result,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            1
        )

    # --------------------------------
    # Detect Suspicious Region
    # --------------------------------

    # At least 5 good matches

    if len(good_matches) >= 5:

        xs = [
            point[0]
            for point in suspicious_points
        ]

        ys = [
            point[1]
            for point in suspicious_points
        ]

        x_min = max(
            min(xs) - 20,
            0
        )

        y_min = max(
            min(ys) - 20,
            0
        )

        x_max = min(
            max(xs) + 20,
            image.shape[1] - 1
        )

        y_max = min(
            max(ys) + 20,
            image.shape[0] - 1
        )

        # Draw rectangle

        cv2.rectangle(
            result,
            (x_min, y_min),
            (x_max, y_max),
            (0, 0, 255),
            2
        )

        cv2.putText(
            result,
            "Suspicious Region",
            (
                x_min,
                max(y_min - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

        print(
            "Suspicious Region : DETECTED"
        )

        return result, True

    else:

        print(
            "Suspicious Region : NOT DETECTED"
        )

        return result, False