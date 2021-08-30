# Installation

You need Python 3 installed with the libraries `pytorch`, `opencv` and `mediapipe`.

    git clone https://github.com/indiana-dev/ASL-Alphabet
    cd ASL-Alphabet
    python3 asl_demo.py

# The Project

During my fourth year of engineering school, I had the opportunity to make an internship inside the Fablab of Capgemini's Applied Innovation Exchange. My final project was to build a neural network capable of recognizing the American Sign Language's alphabet in real time through a webcam input.

All the project is made with Python 3. The main libraries that I used are :

- `pytorch` to build and train the neural network
- `opencv` to capture and process the webcam's frames
- `mediapipe` to detect hands and retrieve the position of their landmarks

`mediapipe` is a neural network library specialized in different body detection solutions like pose detection and face detection. I use it in order to detect hands on the frame, and it outputs a list of 21 landmarks representing the position in 3D of key points of the hand (wrist, fingers, articulations).

I then used `pytorch` to build the neural network that has an input size of 63 (21 landmarks * 3 dimension) and an output size of 26 (all the letters of the alphabet).

# Training 

I created my own dataset using a custom script capturing all the landmarks for each frame and saving them in a json file. At the end, I had 2,000 examples for each of the 26 signs for a result of 54,000 examples. After some data augmentation (vertical mirror to include left-handed people and random noise), I got to around 200,000 examples and started the training. The results of the testing loss and accuracy are shown below : 

<div class="image-list" style="text-align:center;">
  <img class="img-expand" src="assets/loss0.png" width="400px" height=""/> 
  <img class="img-expand" src="assets/acc0.png" width="400px" height=""/> 
</div>

# Demonstration

I made a script that use the trained model in order to predict the signs that are shown through the webcam. Multiple hands can be detected simultaneously and the prediction for each sign is shown next to it on the output frame along with the prediction confiance level.

I also added a feature that writes a sequence of signs as letters on the top left corner of the screen in order for the user to form simple words. Two 'delete' and 'reset' buttons can also be activated when a hand is detected inside their bounding box to have a better control over the sequence of letters.


An example of all of these features can be shown in [this youtube video](https://www.youtube.com/watch?v=XoeKoQWkRt0)