###
### All the content of this file is under the property 
### of Capgemini's Applied Innovation Exchange.
###
### Author : Alexandre Bizord
### Date : 2021
###

import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import time

class SentenceManager:
    def __init__(self):
        self.sentence = ''
        self.currentLetter = None
        self.startTime = None

        self.reset = None
        self.back = None

    def lastLetter(self):
        return self.sentence[-1] if len(self.sentence) > 0 else None
###

custom_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
activation_delay = 500

class ASLNetwork(nn.Module):
    def __init__(self, inputSize, outputSize):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(inputSize, 128),
            nn.Sigmoid(),
            nn.Linear(128, outputSize),
        )
    
    def forward(self, x):      
        return self.architecture(x)

# Load trained model
aslnet = ASLNetwork(63, len(custom_classes))
checkpoint = torch.load('asl_model', map_location=torch.device('cpu'))
aslnet.load_state_dict(checkpoint['model_state_dict'])

###

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def remap_landmarks(landmarks):
  minX = minY = minZ = float('inf')
  maxX = maxY = maxZ = float('-inf')

  for i in range(len(landmarks)):
    if i%3 == 0:
      minX = min(minX, landmarks[i])
      maxX = max(maxX, landmarks[i])
    elif i%3 == 1:
      minY = min(minY, landmarks[i])
      maxY = max(maxY, landmarks[i])
    else:
      minZ = min(minZ, landmarks[i])
      maxZ = max(maxZ, landmarks[i])

  for i in range(len(landmarks)):
    if i%3 == 0:
      landmarks[i] = lerp(landmarks[i], minX, maxX, 0, 1)
    elif i%3 == 1:
      landmarks[i] = lerp(landmarks[i], minY, maxY, 0, 1)
    else:
      landmarks[i] = lerp(landmarks[i], minZ, maxZ, 0, 1)

class HandData:
    def __init__(self):
        self.landmarks = []
    
    def addLandmarks(self, landmarks):
        for lm in landmarks:
            # x = lerp(lm[0], minX, maxX, 0, 1)
            # y = lerp(lm[1], minY, maxY, 0, 1)
            # z = lerp(lm[2], minZ, maxZ, 0, 1)
            self.landmarks.append((lm[0], lm[1], lm[2]))
        # self.landmarks.append((x, y,))


    def toTensor(self):
        flat_landmarks = [item for sublist in self.landmarks for item in sublist]
        remap_landmarks(flat_landmarks)   
        return torch.as_tensor(flat_landmarks)    

def lerp(n, start1, stop1, start2, stop2):
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2

def main():
    sm = SentenceManager()
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()

        frame = cv2.flip(frame, 1)

        # Dimensions of the image
        h, w, c = frame.shape
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameRGB.flags.writeable = False

        # Try to detect hands
        results = hands.process(frameRGB)
        hand_landmarks = results.multi_hand_landmarks
        all_landmarks = []

        # If a hand is detected on screen
        if hand_landmarks:
            # Loop through each hand
            for hlm in hand_landmarks:
                current_hand = HandData()

                minX = minY = minZ = float('inf')
                maxX = maxY = maxZ = float('-inf')

                landmarks = []

                # Loop through each landmark
                for lm in hlm.landmark:
                    mp_drawing.draw_landmarks(frame, hlm, mpHands.HAND_CONNECTIONS)
                    minX = max(min(minX, lm.x), 0)
                    maxX = max(maxX, lm.x)
                    minY = max(min(minY, lm.y), 0)
                    maxY = max(maxY, lm.y)
                    minZ = max(min(minZ, lm.z), 0)
                    maxZ = max(maxZ, lm.z)
                    landmarks.append((lm.x, lm.y, lm.z))
                    all_landmarks.append(lm)
                
                # Calculate the hand bounding box
                pt1, pt2 = get_hand_bounding_rect(minX, minY, maxX, maxY, w, h)

                # Feed forward
                current_hand.addLandmarks(landmarks)
                input = current_hand.toTensor()
                output = aslnet(input)
                softmax = torch.nn.Softmax(dim=0)
                output = softmax(output)
                
                # Retrieve the predicted class and trust factor
                class_index = output.argmax()
                predicted_class = custom_classes[class_index]
                prediction_trust = output[class_index].item()

                now = time.time()

                if sm.currentLetter is not predicted_class:
                    sm.currentLetter = predicted_class
                    sm.startTime = now
                elif sm.lastLetter() == sm.currentLetter and now - sm.startTime > 3*activation_delay/1000 or sm.lastLetter() != sm.currentLetter and now - sm.startTime > activation_delay/1000:
                    sm.sentence += sm.currentLetter
                    print('Sentence: {}'.format(sm.sentence))
                    sm.currentLetter = None

                # Draw the bounding box
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

                # Draw the predicted label
                if sm.reset == None and sm.back == None:
                    draw_label(frame, pt1, predicted_class, prediction_trust)

        # Display the resulting frame
        manage_reset(frame, w, h, all_landmarks, sm)
        manage_back(frame, w, h, all_landmarks, sm)
        draw_sentence(frame, sm.sentence)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def get_hand_bounding_rect(minX, minY, maxX, maxY, w, h):
    minX *= w
    maxX *= w
    minY *= h
    maxY *= h

    maxSize = max(maxX - minX, maxY - minY)/2
    x = (minX + maxX) / 2
    y = (minY + maxY) / 2

    return (int(x - maxSize), int(y - maxSize)), (int(x + maxSize), int(y + maxSize))

def crop_hand(frame, pt1, pt2):
    hand = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]].copy()
    hand = cv2.resize(hand, (32, 32), interpolation = cv2.INTER_AREA)

    return hand

def draw_label(frame, pt1, label, trust):
    label = "Prediction: {} ({:.2f}%)".format(label, trust*100)
    text_font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = 1
    text_thickness = 2
    text_padding = 10

    col_g = lerp(trust, 0.7, 1, 0, 255)
    col_r = 255 - col_g
    color_pred = (0, col_g, col_r)

    (text_w, text_h), text_baseline = cv2.getTextSize(label, text_font, text_size, text_thickness)
    text_baseline -= 2
    text_origin = (pt1[0], pt1[1] + text_padding - text_baseline)
    text_end = (pt1[0] + text_w + text_padding * 2, pt1[1] - text_h - text_padding - text_baseline)

    cv2.rectangle(frame, text_origin, text_end, color_pred, cv2.FILLED)
    cv2.putText(frame, label, (pt1[0] + text_padding, pt1[1] - text_baseline), text_font, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)
              
def draw_sentence(frame, sentence):
    text_size = 2
    text_thickness = 2
    (text_w, text_h), text_baseline = cv2.getTextSize(sentence, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
    cv2.rectangle(frame, (0, 0), (text_w+2, text_h+2), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, sentence, (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

def manage_reset(frame, w, h, landmarks, sm):
    size = int(w * 0.1)
    xMin = 1 - size/w
    xMax = 1
    yMin = 0
    yMax = size/h

    activated = False
    for lm in landmarks:
        if lm.x > xMin and lm.x < xMax and lm.y > yMin and lm.y < yMax:
            activated = True
            if sm.reset == None:
                sm.reset = time.time()
            break

    if not activated and sm.reset != None:
        sm.reset = None

    if activated:
        sm.startTime = None
        sm.currentLetter = None
    
    progress = int(lerp(time.time() - sm.reset, 0, 0.75, 0, size)) if activated else 0

    if activated and progress >= size:
        sm.sentence = ''
        sm.reset = None

    cv2.rectangle(frame, (w - size, 0), (w - size + progress, size), (50, 50, 255), cv2.FILLED)
    cv2.rectangle(frame, (w - size, 0), (w, size), (0, 0, 255), thickness=4)

    text_size = 1
    text_thickness = 2
    (text_w, text_h), text_baseline = cv2.getTextSize('Reset', cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
    cv2.putText(frame, 'Reset', (int(w-size/2-text_w/2), int(size/2 + text_h/2)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

def manage_back(frame, w, h, landmarks, sm):
    size = int(w * 0.1)
    xMin = 1 - 3 * size/w
    xMax = 1 - 2* size/w
    yMin = 0
    yMax = size/h

    activated = False
    for lm in landmarks:
        if lm.x > xMin and lm.x < xMax and lm.y > yMin and lm.y < yMax:
            activated = True
            if sm.back == None:
                sm.back = time.time()
            break

    if not activated and sm.back != None:
        sm.back = None

    if activated:
        sm.startTime = None
        sm.currentLetter = None
    
    progress = int(lerp(time.time() - sm.back, 0, 0.75, 0, size)) if activated else 0

    if activated and progress >= size:
        sm.sentence = sm.sentence[:-1]
        sm.back = None

    cv2.rectangle(frame, (w - 3*size, 0), (w - 3*size + progress, size), (50, 50, 255), cv2.FILLED)
    cv2.rectangle(frame, (w - 3*size, 0), (w - 2*size, size), (0, 0, 255), thickness=4)

    text_size = 1
    text_thickness = 2
    (text_w, text_h), text_baseline = cv2.getTextSize('Delete', cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
    cv2.putText(frame, 'Delete', (int(w-size*2.5-text_w/2), int(size/2 + text_h/2)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

if __name__ == '__main__':
    main()