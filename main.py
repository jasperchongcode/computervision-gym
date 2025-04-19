import cv2
import sys
import mediapipe as mp
import time
import numpy as np

# 0 for webcam
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

source = cv2.VideoCapture(s)

win_name = "Camera Preview"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

desired_landmarks = [28, 26, 24, 27, 25, 23]

left_thigh = [23, 25]
right_thigh = [24, 26]

left_leg = [23, 25, 27]
right_leg = [24, 26, 28]

reps = 0
prev_was_rep = False  # so you dont overcount reps

confidence_threshold = 0.9
depth_y_delta = 45

# STANDING_X_WIDTH = 10  # pixel width for when you are standing upright
# CALIBRATED_Y = None  # this will be calibrated when standing upright


def angle_between(a, b, c):
    ba, bc = np.array(a)-np.array(b), np.array(c)-np.array(b)
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0)))


def calculate_angle(pose_landmarks_list, width, height):
    output = [-1, -1]
    if (pose_landmarks_list[left_leg[0]].visibility >= confidence_threshold
        and pose_landmarks_list[left_leg[1]].visibility >= confidence_threshold
            and pose_landmarks_list[left_leg[2]].visibility >= confidence_threshold):
        # left leg angle
        output[0] = angle_between(
            *[[int(pose_landmarks_list[left_leg[i]].x*width), int(pose_landmarks_list[left_leg[i]].y*height)] for i in range(3)])

    if (pose_landmarks_list[right_leg[0]].visibility >= confidence_threshold
        and pose_landmarks_list[right_leg[1]].visibility >= confidence_threshold
            and pose_landmarks_list[right_leg[2]].visibility >= confidence_threshold):
        output[1] = angle_between(
            *[[int(pose_landmarks_list[right_leg[i]].x*width), int(pose_landmarks_list[right_leg[i]].y*height)] for i in range(3)])

    return output


def at_depth(pose_landmarks_list, image_height) -> bool:
    """Detect depth if there is enough confidence on a thigh, and look for difference in height between points

    return True if confident at depth, False otherwise"""
    # if we have either the left and right leg at above confidence threshold
    # check left leg
    if pose_landmarks_list[left_thigh[0]].visibility >= confidence_threshold and pose_landmarks_list[left_thigh[1]].visibility >= confidence_threshold:
        # therefore we have left thigh at good confidence:
        # check for depth
        if pose_landmarks_list[left_thigh[1]].y*image_height - pose_landmarks_list[left_thigh[0]].y*image_height <= depth_y_delta:
            return True

    if pose_landmarks_list[right_thigh[0]].visibility >= confidence_threshold and pose_landmarks_list[right_thigh[1]].visibility >= confidence_threshold:
        # therefore we have left thigh at good confidence:
        # check for depth
        if pose_landmarks_list[right_thigh[1]].y*image_height - pose_landmarks_list[right_thigh[0]].y*image_height <= depth_y_delta:
            return True

    return False


prevtime = time.time()  # for fps

while cv2.waitKey(1) != 27:  # esc
    # get current frame
    has_frame, frame = source.read()
    if not has_frame:
        break

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    frame = cv2.flip(frame, 1)
    # process results
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    # render
    if results.pose_landmarks:
        mpDraw.draw_landmarks(
            frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

        # print(str(results.visibility))
        pose_landmarks_list = results.pose_landmarks.landmark

        for landmark in desired_landmarks:
            landmark = pose_landmarks_list[landmark]
            x = landmark.x * frame_width
            y = landmark.y * frame_height
            confidence = landmark.visibility

            coords = (int(x), int(y))

            color = (0, 255, 0) if confidence >= confidence_threshold else (
                0, 165, 255)

            cv2.circle(
                frame, coords, 8, color, -1, cv2.FILLED)
            cv2.putText(frame, str(round(confidence, 3)), (coords[0], coords[1]+4),
                        cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 2)
        # check the angle:
        angles = calculate_angle(
            pose_landmarks_list, frame_width, frame_height)
        for i, angle in enumerate(angles):
            if angle != -1:  # if valid angle, render
                if i == 0:
                    coords = (int(pose_landmarks_list[left_leg[1]].x*frame_width)+5, int(
                        pose_landmarks_list[left_leg[1]].y*frame_height))
                else:
                    coords = (int(pose_landmarks_list[right_leg[1]].x*frame_width)+5, int(
                        pose_landmarks_list[right_leg[1]].y*frame_height))

                cv2.putText(frame, f"{round(angle, 0)}°", coords,
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

        # calibrates leg length when standing upright
        # set_calibrated_y(pose_landmarks_list)
        # check depth
        if at_depth(pose_landmarks_list, frame_height):
            cv2.putText(frame, "DEPTH DETECTED!!!", (100, int(frame_height/2)),
                        cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 10)
            if not prev_was_rep:
                reps += 1
                prev_was_rep = True
        else:
            prev_was_rep = False

    # get performance
    frameTime = time.time()
    fps = 1/(frameTime-prevtime)

    prevtime = frameTime

    cv2.putText(frame, f"FPS: {int(fps)}", (2, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.putText(frame, f"Reps: {reps}", (100, 15),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow(win_name, frame)

source.release()
cv2.destroyWindow(win_name)
