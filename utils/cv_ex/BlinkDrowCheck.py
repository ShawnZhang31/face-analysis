import time
import sys

import cv2
import numpy as np
from abc import ABC

class BlinkDrowyCheck(ABC):
    def __init__(self, spf:float, blinkTime:float = 0.2, drowsyTime:float = 1.0, eyeClosedThresh:float = 0.33, leftEyeIndex = [36, 37, 38, 39, 40, 41], rightEyeIndex = [42, 43, 44, 45, 46, 47]):
        """
        check blink and drowy using the landmarks of face. When the drowy status is checked, self.drowsy will be 1, else will be 0.
        Args:
            spf (float): seconds per frame, using for calculating blink time and drowy time on different computer because that the different comptuer has different computer performances. So we must convert the blink time and drowy time to frames.
            blinkTime (float): the duration time of blink. Default is 0.2s
            drowsyTime: (float): the duration time of drowy. Default is 1.0s
            eyeClosedThresh (float): the threshold for judging that the eye is open or closed. Default is 0.33
            leftEyeIndex (array): the landmarks of lef eye. Default is the 68 points shape of dlib. Default is [36, 37, 38, 39, 40, 41]
            rightEyeIndex (array): the landmarks of right eye. Default is the 68 points shape of dlib. Default is [42, 43, 44, 45, 46, 47]
        """
        self.spf = spf      # seconds per frame
        self.blinkCount = 0     # count the number of blinks
        self.drowsy = 0         # is drowy or not, 1->drowsy, 0-> not drowsy
        self.state = 0          # the finite state machine for judging is drowsy or not
        self.blinkTime = blinkTime      # the duration time of blink
        self.drowsyTime = drowsyTime    # the duration time of blink
        self.falseBlinkLimit = self.blinkTime/self.spf      # convert the duration time of blink to blink frames, if bigger than this value is blink, or is not
        self.drowsyLimit = self.drowsyTime/self.spf         # convert the duration time of drowsy to drowsy frames, if bigger than thie value is drowsy, or is not
        self.eyeClosedThresh = eyeClosedThresh      # the threshold for judging that the eye is open or closed, when the eye is open that the value will be lager than eyeClosedThresh, vice versa
        self.leftEyeIndex = leftEyeIndex    # the landmarks of lef eye. Default is the 68 points shape of dlib
        self.rightEyeIndex = rightEyeIndex      # the landmarks of right eye. Default is the 68 points shape of dlib
    
    def checkEyeStatus(self, frame, landmarks, eyeClosedThresh:float = 0.33):
        """
        check the status of eyes
        Args:
            frame: the detected image. Generate mask using the detected image.
            landmarks: the face landmarks of the detected face
            eyeClosedThresh (float): the threshold for judging that the eye is open or closed. Default is 0.33
        Returns:
            return the status of the eye. 1 -> opened, 0 -> closed
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
        check the face is drowsy or not.
        """
        eyeStatus = self.checkEyeStatus(frame, landmarks, eyeClosedThresh= self.eyeClosedThresh)
        self.checkBlinkStatus(eyeStatus)