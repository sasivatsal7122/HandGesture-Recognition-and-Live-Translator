import joblib


import pandas as pd
import numpy as np
import mediapipe as mp
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


""" Loading Pre-trained Models """

# classifier = joblib.load('Trained_models/knn.pkl')
# classifier = joblib.load('Trained_models/logisticreg.pkl')
# classifier = joblib.load('Trained_models/gnb.pkl')
# classifier = joblib.load('Trained_models/svm.pkl')
# classifier = joblib.load('Trained_models/decisiontree.pkl')
classifier = joblib.load('Trained_models/randomforest.pkl')


dataset = pd.read_csv('mcoords_damta.csv')
X_split = dataset.iloc[:, 1:].values
Y_split = dataset.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X_split, Y_split, test_size=0.33)
scaler = StandardScaler().fit(X_train)



exo_landmark = mp.solutions.drawing_utils
exo_landmark_hands = mp.solutions.hands

model = exo_landmark_hands.Hands(
    max_num_hands = 1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flipping the image horizontally for a later selfie-view display, and converting
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, marking the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = model.process(image)

    # Drawing the hand annotations on the image.
    
    # resetiing writeable is True
    image.flags.writeable = True
    
    #converting the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            coords = hand_landmarks.landmark
            exo_landmark.draw_landmarks(image, hand_landmarks, exo_landmark_hands.HAND_CONNECTIONS)
            coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
            coords = scaler.transform([coords])

            predicted = classifier.predict(coords)

        # Defining the Status Box
        cv2.rectangle(image, (0,0), (160, 60), (245, 90, 16), -1)

        # Displaying Class
        cv2.putText(image, 'Predicted Letter'
                    , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(predicted[0])
                    , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Sign Translator', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
