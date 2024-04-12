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

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Create a window and set its size to half the screen
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
window_width = int(screen_width / 2)
window_height = int(screen_height / 2)

# Create a canvas to display the Mediapipe Pose output
canvas = tk.Canvas(root, width=window_width, height=window_height)
canvas.pack()

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
    # cmd_str4 = 'ffmpeg -ss 00:01:00 -to 00:02:00 -i youtube.mp4 -c copy video.mp4'
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
                
def instructor(window_width, window_height, event, queue):
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
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)   
            #  passing the modified connections list   
            mp_drawing.draw_landmarks(image, results.pose_landmarks, connections = custom_connections, landmark_drawing_spec = custom_style)

            # Resize the frame to half its size
            frame = cv2.resize(image, (int(window_width), int(window_height)))

            # calculate angles between landmarks

            # queue.put('Hello from Process 1')
            # event.set()
            # event.clear()

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Instructor Video', cv2.flip(frame, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    # Release the webcam, Mediapipe Pose
    cap.release()
    pose.close()
    # close all the opened windows
    cv2.destroyAllWindows()

def user(event, queue):
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
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
            mp_drawing.draw_landmarks(image, results.pose_landmarks, connections = custom_connections, landmark_drawing_spec = custom_style)

            # Resize the frame to half its size
            frame = cv2.resize(image, (window_width, window_height))

            # event.wait()
            # message = queue.get()
            # print(message)

            # check landmark angles and determine synchronization score with entirety of angles of instructor
                # if timer is a multiple of 4, and a certain angle is more than 20 degrees diff with instructor, then output voice
                # and highlight the body part that is very different from instructor. Also get screengrab of instructor and user
                # with annotations and paste to pdf document

                # if timer is multiple of 1, save the current synchronization score to an array to calculate overall synchronization score at 
                # the very end
            
            ##### resize the instructor video and user video to each fill half of the screen ####

            # Flip the image horizontally for a selfie-view display.
            cv2.imshow('Personal Video', cv2.flip(frame, 1))
            cv2.moveWindow('Personal Video', window_width, -55)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

    # close all the opened windows
    cv2.destroyAllWindows()

if __name__ =="__main__":
    # get_youtube()

    event = multiprocessing.Event()
    queue = multiprocessing.Queue()
    # creating thread
    p1 = multiprocessing.Process(target=instructor, args=(window_width, window_height, event, queue))
    p2 = multiprocessing.Process(target=user, args=(event, queue))
    # starting process 1 and 2
    p1.start()
    p2.start()
    # wait until processes 1 (instructor video) and 2 are finished
    p1.join()
    p2.join()
    # both processes finished
    print("Congrats! Yoga Session Completed")