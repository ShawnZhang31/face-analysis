import time
import sys

import cv2
import numpy as np
from abc import ABC

class BlinkDrowyCheck(ABC):
    def __init__(self, spf:float = 1.0, blinkCount:int = 0, drowsy:int = 0, state:int = 0, blinkTime:float = 0.2, drowsyTime:float = 1.0, eyeClosedThresh:float = 0.33, leftEyeIndex = [36, 37, 38, 39, 40, 41], rightEyeIndex = [42, 43, 44, 45, 46, 47]):
        """
        docstring
        """
        self.spf = spf
        self.blinkCount = blinkCount
        self.drowsy = drowsy
        self.state = state
        self.blinkTime = blinkTime
        self.drowsyTime = drowsyTime
        self.falseBlinkLimit = self.blinkTime/self.spf
        self.drowsyLimit = self.drowsyTime/self.spf
        self.eyeClosedThresh = eyeClosedThresh
        self.leftEyeIndex = leftEyeIndex
        self.rightEyeIndex = rightEyeIndex
    
    def checkEyeStatus(self, frame, landmarks, eyeClosedThresh:float = 0.33):
        """
        docstring
        """
        # create a black image to be used as mask for the eyes
        mask = np.zeros(frame.shape[:2], dtype=np.float32)

        # create a convex hull for using the points of the left and right eyes
        hullLeftEye = []
        for i in range(0, len(self.leftEyeIndex)):
            hullLeftEye.append((landmarks[self.leftEyeIndex[i]][0], landmarks[self.leftEyeIndex[i]][1]))
        cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

        hullRightEye = []
        for i in range(0, len(self.rightEyeIndex)):
            hullRightEye.append((landmarks[self.rightEyeIndex[i]][0], landmarks[self.rightEyeIndex[i]][1]))
        cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

        #Find the length between the corners of the eye and find the white pixels from the mask obtained above. Use the length of left eye to normalize the area under the eyelids
        # find the distance betweent the tips of the left eye
        lenLeftEyeX = landmarks[self.leftEyeIndex[3]][0] - landmarks[self.leftEyeIndex[0]][0]
        lenRightEyeY = landmarks[self.leftEyeIndex[3]][1] - landmarks[self.leftEyeIndex[0]][1]

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

    def checkBlinkStatus(self, eyeStatus):
        """
        simple finite state machine to keep track of the blinks. we can change the behaviour as needed
        """

        # open state and false blink state
        if (self.state >= 0 and self.state <=self.falseBlinkLimit):
            # if eye is open and then stay in this state
            if (eyeStatus):
                self.state = 0
            # else go to the next state
            else:
                self.state += 1
        elif (self.state > self.falseBlinkLimit and self.state <= self.drowsyLimit):
            if (eyeStatus):
                self.state = 0
                self.blinkCount += 1
            else:
                self.state += 1
        else:
            if (eyeStatus):
                self.state = 0
                self.blinkCount += 1
                self.drowsy = 0
            else:
                self.drowsy = 1
    

    def check(self, frame, landmarks):
        """
        docstring
        """
        eyeStatus = self.checkEyeStatus(frame, landmarks, eyeClosedThresh= self.eyeClosedThresh)
        self.checkBlinkStatus(eyeStatus)