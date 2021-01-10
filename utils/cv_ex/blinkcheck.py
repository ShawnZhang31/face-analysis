from os import makedev
import time
import sys
import numpy as np
import cv2
import dlib
from numpy.lib.twodim_base import eye

predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# dlib points for eyes:
leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

# variable for calculating FPS
blinkCount = 0
drowsy = 0
state = 0
blinkTime = 0.2     # 200 ms
drowsyTime = 1.0    # 1000 ms

def checkEyeStatus(frame, landmarks, eyeClosedThresh=0.33):
    """check the eye status, if the eye is open, this func will return 1, or it will return 0.

    Args:
        frame (np.array): Image using for checking
        landmarks (array): facial landmarks
        eyeClosedThresh (float, optional): if normalized eye areas is smaller than eyeClosedThresh will be Closed, or Opened. Defaults to 0.43.

    Returns:
        int: 1 means the eye is opened, and 0 means the eye is closed.
    """
    # create a black image to be used as mask for the eyes
    mask = np.zeros(frame.shape[:2], dtype=np.float32)

    # create a convex hull for using the points of the left and right eyes
    hullLeftEye = []
    for i in range(0, len(leftEyeIndex)):
        hullLeftEye.append((landmarks[leftEyeIndex[i]][0], landmarks[leftEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = []
    for i in range(0, len(rightEyeIndex)):
        hullRightEye.append((landmarks[rightEyeIndex[i]][0], landmarks[rightEyeIndex[i]][1]))
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    #Find the length between the corners of the eye and find the white pixels from the mask obtained above. Use the length of left eye to normalize the area under the eyelids
    # find the distance betweent the tips of the left eye
    lenLeftEyeX = landmarks[leftEyeIndex[3]][0] - landmarks[leftEyeIndex[0]][0]
    lenRightEyeY = landmarks[leftEyeIndex[3]][1] - landmarks[leftEyeIndex[0]][1]

    lenLeftEyeSquare = lenLeftEyeX**2 + lenRightEyeY**2

    # find the area of the eye region
    eyeRegionCount = cv2.countNonZero(mask)

    # normalize the area by the lenght of eye
    # the threshold will not work without the normalization
    # the same amount of eye opening will have more area if it is close to the camera
    normalizedCount = eyeRegionCount/np.float32(lenLeftEyeSquare)

    eyeStatus =1 # 1 -> Opened, 0 -> Closed
    
    if normalizedCount < eyeClosedThresh:
        eyeStatus = 0

    # cv2.putText(mask, str(eyeStatus), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.0, 255)
    # cv2.imshow("mask", mask)

    return eyeStatus

    
    # cv2.waitKey(1)

def checkBlinkStatus(eyeStatus, falseBlinkLimit, drowsyLimit):
    """
    simple finite state machine to keep track of the blinks. we can change the behaviour as needed
    """
    global state, blinkCount, drowsy

    # open state and false blink state
    if (state >= 0 and state <=falseBlinkLimit):
        # if eye is open and then stay in this state
        if (eyeStatus):
            state = 0
        # else go to the next state
        else:
            state += 1
    elif (state > falseBlinkLimit and state <= drowsyLimit):
        if (eyeStatus):
            state = 0
            blinkCount += 1
        else:
            state += 1
    else:
        if (eyeStatus):
            state = 0
            blinkCount += 1
            drowsy = 0
        else:
            drowsy = 1



if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        
        rects = face_detector(frame, 0)
        if len(rects) > 0:
            for face in rects:
                points = []
                [points.append((p.x, p.y)) for p in predictor(frame, face).parts()]
                eyeStatus = checkEyeStatus(frame, points)

        k = cv2.waitKey(1)
        if k == 27:
            break
