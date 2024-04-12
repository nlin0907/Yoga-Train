import cv2
import mediapipe as mp
import numpy as np
import subprocess
import sys
import multiprocessing
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import tkinter as tk
from PIL import Image, ImageTk
import math
from scipy import spatial
import time
# import pyttsx3
import pyautogui
import xlsxwriter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def get_youtube():
    # Remove the original video and download a new one
    cmd_str1 = 'rm -rf youtube.mp4'
    cmd_str2 = 'rm -rf video.mp4'
    # download the youtube with the given ID
    # https://www.youtube.com/watch?v=$YOUTUBE_ID
    # example: https://www.youtube.com/watch?v=KYbFGbLPlb0
    cmd_str3 = 'youtube-dl -f \'bestvideo[ext=mp4]\' --output "youtube.%(ext)s" \'' +  sys.argv[1] + '\''
    # cut the first 40 seconds
    # The timing format is: hh:mm:ss
    # cmd_str4 example: 'ffmpeg -ss 00:01:00 -to 00:02:00 -i youtube.mp4 -c copy video.mp4'
    cmd_str4 = 'ffmpeg -ss ' + sys.argv[2] + ' -to ' + sys.argv[3] + ' -i youtube.mp4 -c copy video.mp4'
    subprocess.run(cmd_str1, shell=True)
    subprocess.run(cmd_str2, shell=True)
    subprocess.run(cmd_str3, shell=True)
    subprocess.run(cmd_str4, shell=True)

# https://stackoverflow.com/questions/75365431/mediapipe-display-body-landmarks-only
# exclude some landmarks 
custom_style = mp_drawing_styles.get_default_pose_landmarks_style()
custom_connections = list(mp_pose.POSE_CONNECTIONS)

# list of landmarks to exclude from the drawing
excluded_landmarks = [
    PoseLandmark.LEFT_EYE, 
    PoseLandmark.RIGHT_EYE, 
    PoseLandmark.LEFT_EYE_INNER, 
    PoseLandmark.RIGHT_EYE_INNER, 
    PoseLandmark.LEFT_EAR,
    PoseLandmark.RIGHT_EAR,
    PoseLandmark.LEFT_EYE_OUTER,
    PoseLandmark.RIGHT_EYE_OUTER,
    PoseLandmark.NOSE,
    PoseLandmark.MOUTH_LEFT,
    PoseLandmark.MOUTH_RIGHT,
    PoseLandmark.RIGHT_PINKY,
    PoseLandmark.LEFT_PINKY,
    PoseLandmark.RIGHT_THUMB,
    PoseLandmark.LEFT_THUMB,
    PoseLandmark.RIGHT_INDEX,
    PoseLandmark.LEFT_INDEX
] 

for landmark in excluded_landmarks:
    # we change the way the excluded landmarks are drawn
    custom_style[landmark] = DrawingSpec(color=(220,209,191), thickness=None, circle_radius = 0) 
    # we remove all connections which contain these landmarks
    custom_connections = [connection_tuple for connection_tuple in custom_connections if landmark.value not in connection_tuple]

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle
                
def instructor(window_width, window_height, queue, event):
    input_video = 'video.mp4'
    cap = cv2.VideoCapture(input_video)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, window_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, window_height)
    if cap.isOpened() == False:
        print("Error opening video stream or file")
        raise TypeError

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            fps = cap.get(cv2.CAP_PROP_FPS)
            # print("fps", fps)

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   
            #  passing the modified connections list   
            mp_drawing.draw_landmarks(image, 
                results.pose_landmarks, 
                connections = custom_connections, 
                connection_drawing_spec = DrawingSpec(color=(0,255,0), thickness=4, circle_radius=2),
                landmark_drawing_spec = custom_style
            )

            # Resize the frame to half its size
            frame = cv2.resize(image, (int(window_width), int(window_height)))

            # calculate angles between landmarks
            if results.pose_landmarks is not None:
                landmarks = results.pose_landmarks.landmark

                left_shoulder = [landmarks[11].x,landmarks[11].y]
                left_elbow = [landmarks[13].x,landmarks[13].y]
                left_wrist = [landmarks[15].x,landmarks[15].y]
                left_hip = [landmarks[23].x,landmarks[23].y]
                left_knee = [landmarks[25].x,landmarks[25].y]
                left_ankle = [landmarks[27].x,landmarks[27].y]

                right_shoulder = [landmarks[12].x,landmarks[12].y]
                right_elbow = [landmarks[14].x,landmarks[14].y]
                right_wrist = [landmarks[16].x,landmarks[16].y]
                right_hip = [landmarks[24].x,landmarks[24].y]
                right_knee = [landmarks[26].x,landmarks[26].y]
                right_ankle = [landmarks[28].x,landmarks[28].y]

                left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                angle_arr_instructor = np.array([left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle, left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle])

                # send message to user
                queue.put(angle_arr_instructor)

            cv2.imshow('Instructor Video', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    # Release the Mediapipe Pose
    cap.release()
    # close all the opened windows
    time.sleep(8)
    cv2.destroyAllWindows()
    event.set()

def user(queue, window_width, window_height, event):
    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)
    similarity_total_arr = []

    workbook = xlsxwriter.Workbook('yoga_train_outputs.xlsx')
    ws = workbook.add_worksheet()

    landmark_dict = ["left shoulder", "right shoulder", "left elbow", "right elbow", "left hip", "right hip", "left knee", "right knee"]

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_number = 0
        img_num = 0
        old_instructor_angles = [0,0,0,0,0,0,0,0]
        first_time = True

        while cap.isOpened() and not event.is_set():
            frame_number += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(image, 
                results.pose_landmarks, 
                connections = custom_connections, 
                connection_drawing_spec = DrawingSpec(color=(0,255,0), thickness=4, circle_radius=2),
                landmark_drawing_spec = custom_style
            )

            # Resize and flip the frame to half its size
            frame = cv2.resize(image, (window_width, window_height))
            frame = cv2.flip(frame, 1)

            message = queue.get()
            # print("message", message)

            if len(message) == 8:
                # calculate angles between landmarks
                if results.pose_landmarks is not None:
                    landmarks = results.pose_landmarks.landmark
                    # find coordinates of landmarks
                    left_shoulder = [landmarks[11].x,landmarks[11].y]
                    left_elbow = [landmarks[13].x,landmarks[13].y]
                    left_wrist = [landmarks[15].x,landmarks[15].y]
                    left_hip = [landmarks[23].x,landmarks[23].y]
                    left_knee = [landmarks[25].x,landmarks[25].y]
                    left_ankle = [landmarks[27].x,landmarks[27].y]

                    right_shoulder = [landmarks[12].x,landmarks[12].y]
                    right_elbow = [landmarks[14].x,landmarks[14].y]
                    right_wrist = [landmarks[16].x,landmarks[16].y]
                    right_hip = [landmarks[24].x,landmarks[24].y]
                    right_knee = [landmarks[26].x,landmarks[26].y]
                    right_ankle = [landmarks[28].x,landmarks[28].y]

                    # left_shoulder = [landmarks[11].x,landmarks[11].y, landmarks[11].z]
                    # left_elbow = [landmarks[13].x,landmarks[13].y, landmarks[13].z]
                    # left_wrist = [landmarks[15].x,landmarks[15].y, landmarks[15].z]
                    # left_hip = [landmarks[23].x,landmarks[23].y, landmarks[23].z]
                    # left_knee = [landmarks[25].x,landmarks[25].y, landmarks[25].z]
                    # left_ankle = [landmarks[27].x,landmarks[27].y, landmarks[27].z]

                    # right_shoulder = [landmarks[12].x,landmarks[12].y, landmarks[12].z]
                    # right_elbow = [landmarks[14].x,landmarks[14].y, landmarks[14].z]
                    # right_wrist = [landmarks[16].x,landmarks[16].y, landmarks[16].z]
                    # right_hip = [landmarks[24].x,landmarks[24].y, landmarks[24].z]
                    # right_knee = [landmarks[26].x,landmarks[26].y, landmarks[26].z]
                    # right_ankle = [landmarks[28].x,landmarks[28].y, landmarks[28].z]

                    # calculate angles between landmarks
                    left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    left_shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
                    left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
                    left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

                    right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
                    right_shoulder_angle = calculate_angle(right_elbow, right_shoulder, right_hip)
                    right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
                    right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)

                    coordinate_arr_user = [left_shoulder, right_shoulder, left_elbow, right_elbow, left_hip, right_hip, left_knee, right_knee]
                    angle_arr_user = np.array([left_shoulder_angle, right_shoulder_angle, left_elbow_angle, right_elbow_angle, left_hip_angle, right_hip_angle, left_knee_angle, right_knee_angle])
                    # Get the absolute difference between arrays
                    diff = np.abs(message - angle_arr_user)
                    idx = np.argmax(diff) 
                    if (50 <= diff[idx] <= 160 and (landmark_dict[idx] == "left shoulder" or landmark_dict[idx] == "right shoulder")) or (diff[idx] >= 30 and not (landmark_dict[idx] == "left shoulder" or landmark_dict[idx] == "right shoulder")):
                    # if diff[idx] >= 70 and diff[idx] <= 170: # thresholds chosen because sometimes mediapipe just doesn't capture the pose even if they are exactly the same
                        # show text telling which body part is incorrect
                        # if body index is actually in frame
                        if int(coordinate_arr_user[idx][0] * window_width) < window_width and  int(coordinate_arr_user[idx][1] * window_height) < window_height:
                            print("angle_arr_user", angle_arr_user)
                            print("message", message)
                            print("is diff", landmark_dict[idx])
                            print("diff", diff)
                            # circle wrong body part with red circle
                            cv2.circle(frame, (abs(frame.shape[1] - int(coordinate_arr_user[idx][0] * frame.shape[1])), int(coordinate_arr_user[idx][1] * frame.shape[0])), 25, (40, 50, 255), 4)
                            # cv2.putText(frame, landmark_dict[idx], (abs(frame.shape[1] - int(coordinate_arr_user[idx][0] * frame.shape[1])), int(coordinate_arr_user[idx][1] * frame.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                            # every 4 seconds, check if there are any incorrect body parts, if there is, grab a screenshot and add to the excel file
                            if frame_number % math.floor(2*fps) == 0:
                                # print("here")
                                # print("instructor", message)
                                # print("user", angle_arr_user)
                                instructor_diff = np.sum(np.abs(message - old_instructor_angles))/8
                                # print("instructor_diff", instructor_diff)
                                if instructor_diff < 50 or first_time:
                                    print("say cheese!")
                                    first_time = False
                                    old_instructor_angles = message
                                    # compare with new_instructor_message = message
                                    # if different by a certain percentage, then dont grab screenshot because instructor is switching between positions

                                    # grab screenshot of frame and paste into pdf document
                                    screenshot = pyautogui.screenshot(region=(0, 100, 2875, 900))
                                    screenshot.save('yt_screenshot' + str(img_num) + '.png')
                                    ws.set_row(img_num, 300)
                                    ws.insert_image("A"+str(img_num + 1), 'yt_screenshot' + str(img_num) + '.png', {'x_scale': 0.4, 'y_scale': 0.4, 'object_position': 1})
                                    img_num += 1

                    angle_arr_user = angle_arr_user.reshape(-1, 1)
                    message = message.reshape(-1, 1)
                    cos_sim = 1 - (spatial.distance.cosine(message, angle_arr_user) * 1.1)
                    cv2.putText(frame, "Similarity: " + str(round((cos_sim * 100),2)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    # mse = np.mean((message - angle_arr_user) ** 2)
                    # score = 1- (math.tanh(0.001 * mse))
                    # cv2.putText(frame, "Similarity: " + str(round(score,4)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    if (frame_number % math.floor(fps) == 0):
                        # if timer is at one second, save cosine similarity between two arrays to calculate overall synchronization at the end
                        similarity_total_arr.append(cos_sim*100)
                        # similarity_total_arr.append(score)

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Personal Video', frame)
            cv2.moveWindow('Personal Video', window_width, -55)
            if cv2.waitKey(5) & 0xFF == 27:
                # Close the workbook
                workbook.close()
                break

    cap.release()
    # close all the opened windows
    time.sleep(3)
    cv2.destroyAllWindows()

    average_similarity = np.mean(similarity_total_arr)
    ws.write(0, 19, "Average similarity: " + str(round(average_similarity,4)))
    # Close the workbook
    workbook.close()

if __name__ =="__main__":
    get_youtube()

    # Create a window and set its size to half the screen
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = int(screen_width / 2)
    window_height = int(screen_height / 2)

    workbook = xlsxwriter.Workbook('yoga_train_outputs.xlsx')
    ws = workbook.add_worksheet()

    queue = multiprocessing.Queue()
    event = multiprocessing.Event()
    # creating thread
    p1 = multiprocessing.Process(target=instructor, args=(window_width, window_height, queue, event))
    p2 = multiprocessing.Process(target=user, args=(queue, window_width, window_height, event))
    # starting process 1 and 2
    p1.start()
    p2.start()
    # wait until processes 1 (instructor video) and 2 are finished
    p1.join()
    print("after p1 join")
    # https://stackoverflow.com/questions/47903791/how-to-terminate-a-multiprocess-in-python-when-a-given-condition-is-met
    # if event.is_set():
    #     p2.terminate()
    p2.join()
    # both processes finished
    print("Congrats! Yoga Session Completed")