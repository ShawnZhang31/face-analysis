from ast import dump
import os
import sys
import time
import numpy as np
import cv2
import dlib
# import facelib
import argparse
import operator

from utils import cv_ex
from utils.cv_ex import BlinkDrowyCheck
from fer import FER
from PIL import ImageFont, ImageDraw, Image
import yaml
import face_recognition

faces_configs_path = "./res/faces"
face_configs_file = "faces.yaml"

# load the known faces yaml config file
with open(os.path.join(faces_configs_path, face_configs_file), 'r') as f:
    faces_known = yaml.load(f, Loader=yaml.SafeLoader)

known_face_encodings = [] # known face feature vectors
known_face_names = [] # known face names

for face in faces_known:
    face_name = face['name']
    images = face['images']
    for image_name in images:
        image_path = os.path.join(faces_configs_path, image_name)
        if os.path.exists(image_path):
            face_image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(face_image)[0]

            
            # save the face feature vectors and names for later face comparing
            known_face_encodings.append(face_encoding)
            known_face_names.append(face_name)
        else:
            print("can't find the file: {}, the program will be exit!".format(image_path))
            sys.exit(1)

# print(known_face_encodings)

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

# [{'box': (146, 120, 114, 114), 'emotions': {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}}]
EMOTION_CN={"angry":"愤怒", "disgust": "厌恶", "fear":"害怕", "happy":"高兴", "sad":"悲伤", "surprise":"惊讶", "neutral":"自然"}

def drawUnicodeText(img, text, org, color=(255, 0, 255, 0), fontpath="./res/fonts/Arial Unicode.ttf", fontsize=32):
    """
    draw Unicode Text on the image
    Args:
        img (np.array): Image will be drawn
        text (str): Text will be drawn
        org (tuple): Text origin point
        color (tuple): The color of the text
        fontpath (str): The font file path
        fontsize (int): The size of font
    Returns:
        np.array, the image has been drawn text
    """
    assert(os.path.exists(fontpath) and fontsize > 0)
    font = ImageFont.truetype(fontpath, fontsize)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(org, text, font=font, fill = color)
    return np.array(img_pil)
    

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

###########################################################
# Calculate the FPS for initialization
# Different computers will have relatively different speeds
# Since all operations are on frame basis
# We want to find how many frames correspond to the blink and drowsy limit

# reading some dummy frames for adjust the sensor to the light
frameCounter = 0
for i in range(100):
    frameCounter += 1
    ret, frame = video_capter.read()
    cv2.flip(frame, 1, frame)
    text_org = (30, 30)
    sub_prex = "." * (frameCounter % 6 + 1)
    frame =  drawUnicodeText(frame, "等待摄像头自动调整"+sub_prex, text_org)
    cv2.imshow("Initializing", frame)
    cv2.waitKey(1)
cv2.destroyWindow("Initializing")

# calculating the params about the computer
totalTime = 0.0
validFrames = 0
dummyFrames = 50
spf = 0 #seconds per frame

frameCounter = 0
while (validFrames < dummyFrames):
    frameCounter += 1
    validFrames += 1
    t = time.time()
    ret, frame = video_capter.read()
    cv2.flip(frame, 1, frame)

    frameScaled = cv2.resize(frame, dsize=None, fx= 1.0 / FACE_DOWNSAMLE_RATIO, fy= 1.0 / FACE_DOWNSAMLE_RATIO)

    faces_emotions = emotion_detector.detect_emotions(frameScaled)
    # [{'box': (146, 120, 114, 114), 'emotions': {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}}]

    # print(faces_emotions)

    timeFaces = time.time() - t
    # if face not detected then dont add this time to the calculation
    if len(faces_emotions) == 0:
        validFrames -= 1
    else:
        totalTime += timeFaces

    text_org = (30, 30)
    # sub_prex = "." * (frameCounter % 6 + 1)
    frame =  drawUnicodeText(frame, "正在统计计算机算力性能{}%".format(int((validFrames/dummyFrames)*100)), text_org, color=(0, 255, 255, 0))
    cv2.imshow("Initializing", frame)
    if cv2.waitKey(1) & 0xFF == 27:
            sys.exit()

cv2.destroyWindow("Initializing")

spf = totalTime/dummyFrames
print("Current SPF (seconds per frame) is {:.2f} ms".format(spf*1000))

# drowsyLimit = drowsyTime/spf
# falseBlinkLimit = blinkTime/spf
# print ("drowsyLimit {} ( {:.2f} ms) ,  False blink limit {} ( {:.2f} ms) ".format(drowsyLimit, drowsyLimit*spf*1000, falseBlinkLimit, (falseBlinkLimit+1)*spf*1000))

# create the blinkdrowychecker array
known_face_blinkdrowy_status = []
for face_name in known_face_names:
    blinkchecker = BlinkDrowyCheck(spf=spf, eyeClosedThresh=0.45)
    known_face_blinkdrowy_status.append(blinkchecker)

# unknown face counter
unknown_face_counter = 1

FRAMES_COUNTS = 0
SKIP_FRAMES = 2

frameScaled = None
faces_emotions = None
while (True):
    # Capture frame-by-frame
    ret, frame = video_capter.read()
    cv2.flip(frame, 1, frame)

    

    if FRAMES_COUNTS % SKIP_FRAMES == 0:
        frameScaled = cv2.resize(frame, dsize=None, fx= 1.0 / FACE_DOWNSAMLE_RATIO, fy= 1.0 / FACE_DOWNSAMLE_RATIO)
        faces_emotions = emotion_detector.detect_emotions(frameScaled)
        # [{'box': (146, 120, 114, 114), 'emotions': {'angry': 0.22, 'disgust': 0.0, 'fear': 0.03, 'happy': 0.0, 'sad': 0.13, 'surprise': 0.0, 'neutral': 0.62}}]
    
    # convert for face recognition
    # frameScaled_rgb = frameScaled[:, :, ::-1]
    frameScaled_rgb = frame[:, :, ::-1]
    # face_locations = face_recognition.face_locations(frameScaled_rgb) #(top, right, bottom, left)

    # print(faces_emotions)
    # print(face_locations)

    # get all face encodings once time for saving time
    face_detected_locations = []
    for _rect in faces_emotions:
        box = _rect['box'] #(left, top, width, height)
        # we must convert the rect to (top, right, bottom, left)
        # FACE_DOWNSAMLE_RATIO
        # face_detected_locations.append((int(box[1]), int(box[0]+box[2]), int(box[1]+box[3]), int(box[0])))
        face_detected_locations.append((int(box[1]*FACE_DOWNSAMLE_RATIO),
                                        int((box[0]+box[2])*FACE_DOWNSAMLE_RATIO),
                                        int((box[1]+box[3])*FACE_DOWNSAMLE_RATIO),
                                        int(box[0]*FACE_DOWNSAMLE_RATIO)))

    detected_face_encodings = face_recognition.face_encodings(frameScaled_rgb, face_detected_locations)
    # print(len(detected_face_encodings))

    for idx, face in enumerate(faces_emotions):
        # face recognition
        # see if the face is a match for the known faces
        matches = face_recognition.compare_faces(known_face_encodings, detected_face_encodings[idx], 0.4)
        face_name = "未知"

        # print("{} - {}".format(idx, matches))
        # print(known_face_names)

        if True in matches:
            first_match_index = matches.index(True)
            face_name = known_face_names[first_match_index]
        else:
            face_name = "未知-" + str(unknown_face_counter)
            known_face_encodings.append(detected_face_encodings[idx])
            known_face_names.append(face_name)
            unknown_face_counter += 1
            known_face_blinkdrowy_status.append(BlinkDrowyCheck(spf=spf))


        # print(name)

        box = face['box'] #(left, top, width, height)
        box = [int(box[0]*FACE_DOWNSAMLE_RATIO),
                int(box[1]*FACE_DOWNSAMLE_RATIO),
                int((box[0]+box[2])*FACE_DOWNSAMLE_RATIO),
                int((box[1]+box[3])*FACE_DOWNSAMLE_RATIO)]


        # cv2.circle(frame, (box[0], box[1]), 5, (255, 0, 0)) # left-top
        # cv2.circle(frame, (box[2], box[3]), 5, (0, 255, 0)) # right-bottom

        faceImgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faceRect = dlib.rectangle(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        shape = predictor(faceImgGray, faceRect)

        # face landmarks
        landmarks = []
        [landmarks.append((p.x, p.y)) for p in shape.parts()]

        # check blink drowy status
        blinkchecker:BlinkDrowyCheck = known_face_blinkdrowy_status[idx]
        blinkchecker.check(frame, landmarks)

        # if the drowsy is True, draw the face rect with red rect, else green rect
        if blinkchecker.drowsy:
            drawFaceRect(frame, box, (0, 0, 255, 0))
            frame = drawUnicodeText(frame, "!!!检测到打瞌睡!!!", (box[0], box[1]+32), color=(0, 0, 255, 0))
        else:
            drawFaceRect(frame, box)
        

        drawFaceLandmarks(frame, landmarks)

        emotions = face['emotions']
        frame = drawFaceEmotions(frame, box, emotions)

        # draw face name
        frame = drawUnicodeText(frame, face_name, (box[0], box[1]), color=(0, 255, 0, 0))

    # boxes, sorces, landmarks = face_detector.detect_faces(frame)
    # faces, boxes, scores, landmarks = face_detector.detect_align(frame)
    # list_of_emotions, probab = emotion_detector.detect_emotion(faces)
    # print(list_of_emotions)

    # drowsy_array = []
    # [drowsy_array.append(blinkcheck.drowsy) for blinkcheck in known_face_blinkdrowy_status]
    # print(drowsy_array)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    FRAMES_COUNTS += 1

    if FRAMES_COUNTS >=1000:
        FRAMES_COUNTS = 0

    # print(known_face_names)

    # Break the Loop
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.release()
cv2.destroyAllWindows()

