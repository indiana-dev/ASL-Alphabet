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

custom_classes = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
activation_delay = 500

### Hand Data

class HandData:
    def __init__(self):
        self.landmarks = []
    
    def addLandmarks(self, landmarks):
        for lm in landmarks:
            self.landmarks.append((lm[0], lm[1], lm[2]))

    def toTensor(self):
        flat_landmarks = [item for sublist in self.landmarks for item in sublist]
        remap_landmarks(flat_landmarks)   
        return torch.as_tensor(flat_landmarks) 

### Neural Network

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

### Load trained model

aslnet = ASLNetwork(63, len(custom_classes))
checkpoint = torch.load('asl_model', map_location=torch.device('cpu'))
aslnet.load_state_dict(checkpoint['model_state_dict'])

### Initialize mediapipe

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils 

### Main

def main():
    sm = SentenceManager()
    cap = cv2.VideoCapture(0)

    while(True):
        # Capture frame-by-frame
        _, frame = cap.read()
        # Flip the frame vertically
        frame = cv2.flip(frame, 1) 

        # Dimensions of the image
        h, w, c = frame.shape

        # Convert the frame to BGR for Mediapipe
        frameBGR = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frameBGR.flags.writeable = False

        # Try to detect hands
        results = hands.process(frameBGR)
        hand_landmarks = results.multi_hand_landmarks
        current_hand = None

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
                    minX = max(min(minX, lm.x), 0)
                    maxX = max(maxX, lm.x)
                    minY = max(min(minY, lm.y), 0)
                    maxY = max(maxY, lm.y)
                    minZ = max(min(minZ, lm.z), 0)
                    maxZ = max(maxZ, lm.z)
                    landmarks.append((lm.x, lm.y, lm.z))
                
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

                # Update the sentence
                sm.update(predicted_class)

                # Draw the hand's landmarks
                mp_drawing.draw_landmarks(frame, hlm, mpHands.HAND_CONNECTIONS)

                # Draw the bounding box
                cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)

                # Draw the predicted label
                if sm.reset == None and sm.back == None:
                    draw_label(frame, pt1, predicted_class, prediction_trust)

        # Draw the current sentence
        sm.draw_sentence(frame)
        # Draw and manage the buttons
        sm.draw_buttons(frame, w, h, current_hand.landmarks if hand_landmarks and len(hand_landmarks) == 1 else None)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Quit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Returns the bounding rect of the hand in pixel coordinates
# minX, minY, maxX, maxY: current bouding rect in UV coordinates
# w, h: dimension of the frame
def get_hand_bounding_rect(minX, minY, maxX, maxY, w, h):
    minX *= w
    maxX *= w
    minY *= h
    maxY *= h

    maxSize = max(maxX - minX, maxY - minY)/2
    x = (minX + maxX) / 2
    y = (minY + maxY) / 2

    return (int(x - maxSize), int(y - maxSize)), (int(x + maxSize), int(y + maxSize))

# Draw the predicted label on the frame
# pt1: position of the label
# label: predicted letter
# trust: percentage of trust
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

# Linear interpolation
def lerp(n, start1, stop1, start2, stop2):
    return (n - start1) / (stop1 - start1) * (stop2 - start2) + start2

# Remap landmarks to be in the range 0,1
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

### Sentence Manager

class SentenceManager:
    def __init__(self):
        self.sentence = ''
        self.currentLetter = None
        self.startTime = None

        self.reset = None
        self.back = None

    # Retrieve the last letter of the sentence, or None if the sentence is empty
    def lastLetter(self):
        return self.sentence[-1] if len(self.sentence) > 0 else None

    # Update the current sentence
    def update(self, predicted_class):
        now = time.time()

        if self.currentLetter is not predicted_class:
            self.currentLetter = predicted_class
            self.startTime = now
        elif self.lastLetter() == self.currentLetter and now - self.startTime > 3*activation_delay/1000 or self.lastLetter() != self.currentLetter and now - self.startTime > activation_delay/1000:
            self.sentence += self.currentLetter
            self.currentLetter = None
            print('Sentence: {}'.format(self.sentence))

    def update_button(self, button_name, value):
        if (button_name == 'Delete'):
            self.back = value
        else:
            self.reset = value

    def manage_and_draw_button(self, button_name, button, frame, w, h, landmarks):
        # Button size in pixels
        size = int(w * 0.1)
        # Button bounding rect in the range 0,1
        xOffset = (1 if button_name == 'Reset' else 3) * size
        xMin, xMax = 1 - xOffset/w, 1 - (xOffset-size)/w
        yMin, yMax = 0, size/h

        # Detect if a landmark is inside the button's bounding rect
        activated = False

        if landmarks != None:
            for lm in landmarks:
                if lm[0] > xMin and lm[0] < xMax and lm[1] > yMin and lm[1] < yMax:
                    activated = True
                    if button == None:
                        self.update_button(button_name, time.time())
                        button = time.time()
                    break
        
        # Deactivate the button if it was previously activated and isn't anymore
        if not activated and button != None:
            self.update_button(button_name, None)
            button = None

        if activated:
            self.startTime = None
            self.currentLetter = None
        
        progress = int(lerp(time.time() - button, 0, 0.75, 0, size)) if activated else 0

        if activated and progress >= size:
            self.sentence = self.sentence[:-1] if button_name == 'Delete' else ''
            self.update_button(button_name, None)
            button = None

        xMin, xMax, yMin, yMax = int(xMin*w), int(xMax*w), int(yMin*h), int(yMax*h)
        cv2.rectangle(frame, (xMin, yMin), (xMin + progress, yMax), (50, 50, 255), cv2.FILLED)
        cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (0, 0, 255), thickness=4)

        text_size = 1
        text_thickness = 2
        (text_w, text_h), text_baseline = cv2.getTextSize(button_name, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
        cv2.putText(frame, button_name, (int(xMin + (size-text_w)/2), int(size/2 + text_h/2)), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

    # Draw the current sentence
    def draw_sentence(self, frame):
        text_size = 2
        text_thickness = 2
        (text_w, text_h), text_baseline = cv2.getTextSize(self.sentence, cv2.FONT_HERSHEY_SIMPLEX, text_size, text_thickness)
        cv2.rectangle(frame, (0, 0), (text_w+2, text_h+2), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, self.sentence, (0, text_h), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 0, 0), text_thickness, cv2.LINE_AA)

    # Manage and Draw the Delete and Reset buttons
    def draw_buttons(self, frame, w, h, landmarks):
        if landmarks == None:
            self.back = None
            self.reset = None

        self.manage_and_draw_button('Delete', self.back, frame, w, h, landmarks)
        self.manage_and_draw_button('Reset', self.reset, frame, w, h, landmarks)
        
##############

if __name__ == '__main__':
    main()