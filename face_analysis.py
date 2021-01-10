import os
import numpy as np
import cv2
import dlib
# import facelib
import argparse
import operator

from numpy.lib.type_check import imag

from utils import cv_ex
from fer import FER
from PIL import ImageFont, ImageDraw, Image

predictor_path = "./models/shape_predictor_68_face_landmarks.dat"
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

emotion_detector = FER(mtcnn=True)
# from facelib import FaceDetector, EmotionDetector

print("OpenCV Version:", cv2.__version__)
print("dlib Version:", dlib.__version__)


# retina_face_model = "./models/mobilenet0.25_Final.pth"
# age_gender_model = "./models/ShufflenetFull.pth"
# face_expression_model = "./models/Resnet50_Final.pth"
# # face


# face_detector = FaceDetector(name="mobilenet", weight_path=retina_face_model, device='cpu')
# emotion_detector = EmotionDetector(name='resnet34', weight_path=face_expression_model, device='cpu')

video_capter = cv2.VideoCapture(0)
RESIZE_HEIGHT = 240
RESIZE_WIDTH = 320

FACE_DOWNSAMLE_RATIO = 1.0

SKIP_FRAMES = 10
FRAME_COUNTER = 0
# [{'box': (146, 120, 114, 114), 'emotions': {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}}]
EMOTION_CN={"angry":"愤怒", "disgust": "厌恶", "fear":"害怕", "happy":"高兴", "sad":"悲伤", "surprise":"惊讶", "neutral":"自然"}

def drawFaceRect(img, box, color=(0, 255, 0, 0)):
    """Draw Face Rect

    Args:
        img ([np.array]): Image
        box ([array]): Face Rectange
        color (tuple, optional): The color of face rectange. Defaults to (0, 255, 0, 0).
    """
    point1 = (box[0], box[1])   #Top Left Point
    point2 = (box[2], box[3])     # Right Bottom Point
    cv2.rectangle(img, point1, point2, color)

def drawFaceEmotions(img, box, emotions , color=(0, 255, 0, 0), highlight=(0, 0, 255, 0)):
    """Draw face emotion detection results

    Args:
        img ([np.arrag]): image
        box ([array]): face rectangle box
        emotions (dict): face emotions, like: {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}
        color (tuple, optional): face emotion text color. Defaults to (0, 255, 0, 0).
        highlight (tuple, optional): face emotion highlight text color. Defaults to (0, 0, 255, 0).
    """
    # points top right
    point = (box[2], box[1])
    fontFace = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.6
    space = 30
    count = 0
    # dict(sorted(x.items(), key=lambda item: item[1]))
    sorted_emotions = sorted(emotions.items(), key=operator.itemgetter(1), reverse=True)
    # print(sorted_emotions)
    fontpath = "./res/fonts/Arial Unicode.ttf"
    font = ImageFont.truetype(fontpath, 24)
    img_pil = Image.fromarray(img)
    for (key, value) in emotions.items():
        textColor = color
        if value == sorted_emotions[0][1]:
            textColor = highlight
        
        emotion_str = EMOTION_CN[key] + ": " + str(value)
        org = (point[0]+5, point[1]+ 10 + count*space)
        
        # cv2.putText(img, emotion_str, org, fontFace, fontScale, textColor)

        draw = ImageDraw.Draw(img_pil)
        draw.text(org, emotion_str, font=font, fill = textColor)
        count += 1
    
    img = np.array(img_pil, copy=True)
    return img

def drawFaceLandmarks(img, landmarks, color=(0, 255, 0, 0)):
    for p in landmarks:
        cv2.circle(img, p, 1, color)

ret, frame = video_capter.read()
frame_height = frame.shape[0]
frame_width = frame.shape[1]
frame_channel = frame.shape[2]

FACE_DOWNSAMLE_RATIO = frame_height / RESIZE_HEIGHT

while (True):
    # Capture frame-by-frame
    ret, frame = video_capter.read()
    cv2.flip(frame, 1, frame)

    frameScaled = cv2.resize(frame, dsize=None, fx= 1.0 / FACE_DOWNSAMLE_RATIO, fy= 1.0 / FACE_DOWNSAMLE_RATIO)
    faces_emotions = emotion_detector.detect_emotions(frameScaled)
    # [{'box': (146, 120, 114, 114), 'emotions': {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}}]
    for face in faces_emotions:
        box = face['box']
        box = [int(box[0]*FACE_DOWNSAMLE_RATIO),
                int(box[1]*FACE_DOWNSAMLE_RATIO),
                int((box[0]+box[2])*FACE_DOWNSAMLE_RATIO),
                int((box[1]+box[3])*FACE_DOWNSAMLE_RATIO)]

        faceImgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceRect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        shape = predictor(faceImgGray, faceRect)

        # face landmarks
        landmarks = []
        [landmarks.append((p.x, p.y)) for p in shape.parts()]

        drawFaceLandmarks(frame, landmarks)

        drawFaceRect(frame, box)
        emotions = face['emotions']
        frame = drawFaceEmotions(frame, box, emotions)



    # boxes, sorces, landmarks = face_detector.detect_faces(frame)
    # faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    # list_of_emotions, probab = emotion_detector.detect_emotion(faces)
    # print(list_of_emotions)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Break the Loop
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.release()
cv2.destroyAllWindows()

