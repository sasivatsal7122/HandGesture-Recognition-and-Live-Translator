import joblib
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import av
import copy

import cv2
import numpy as np
import mediapipe as mp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

st.set_page_config(page_title="Talku+", page_icon="🤖")

st.title("Real Time Hand Gesture recognition and Live Hand Sign Translator")
st.text("Developed by Team - x")



class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        
        image = frame.to_ndarray(format="bgr24")
        # Loading Pre-trained Models 

        # classifier = joblib.load('Trained_models/knn.pkl')
        # classifier = joblib.load('Trained_models/logisticreg.pkl')
        # classifier = joblib.load('Trained_models/gnb.pkl')
        # classifier = joblib.load('Trained_models/svm.pkl')
        # classifier = joblib.load('Trained_models/decisiontree.pkl')
        classifier = joblib.load('Trained_models/randomforest.pkl')

        # Loading Pre-Trained Scaler to normalize the input values 
        scaler = joblib.load('Trained_models/StandardScaler.pkl')


        # Making the model to make landmarks using built-in mediapipe hand model

        exo_landmark = mp.solutions.drawing_utils
        exo_landmark_hands = mp.solutions.hands

        model = exo_landmark_hands.Hands(
            max_num_hands = 1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
        
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        
        # To improve performance, marking the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = model.process(image)
        
        # Drawing the hand annotations on the image.
    
        # resetiing writeable is True
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                
                coords = hand_landmarks.landmark
                
                exo_landmark.draw_landmarks(debug_image, hand_landmarks, exo_landmark_hands.HAND_CONNECTIONS)
                
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                
                coords = scaler.transform([coords])

                predicted = classifier.predict(coords)
        # Defining the Status Box
                cv2.rectangle(debug_image, (0,0), (160, 60), (245, 90, 16), -1)

                # Displaying Class
                cv2.putText(debug_image, 'Predicted Letter'
                            , (20,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(debug_image, str(predicted[0])
                            , (20,45), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        return av.VideoFrame.from_ndarray(debug_image, format="bgr24")


webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )