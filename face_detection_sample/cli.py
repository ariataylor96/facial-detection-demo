import argparse
import cv2

from time import sleep

import sys


def main(args):
    face_cascades = [
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml"),
        cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml"),
    ]

    RED = (0, 0, 255)
    GREEN = (0, 255, 0)

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        center_point = [i / 2 for i in frame.shape[:2]]
        center_y_bounds, center_x_bounds = [
            (i - i * args.margin, i + i * args.margin) for i in center_point
        ]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []

        for face_cascade in face_cascades:
            faces.extend(
                face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                )
            )

        for (x, y, w, h) in faces:
            color = RED
            face_center_x, face_center_y = (x + (w / 2), (y + (h / 2)))

            if (
                center_x_bounds[0] <= face_center_x
                and face_center_x <= center_x_bounds[1]
                and center_y_bounds[0] <= face_center_y
                and face_center_y <= center_y_bounds[1]
            ):
                color = GREEN

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        sleep(args.sleep)

    video_capture.release()
    cv2.destroyAllWindows()

    return 0


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--margin",
        type=float,
        help="Percentage from the center the face is allowed to be",
        default=0.05,
    )
    parser.add_argument(
        "-s", "--sleep", type=float, help="Time to sleep between each frame", default=0.05
    )
    args = parser.parse_args()

    sys.exit(main(args))
