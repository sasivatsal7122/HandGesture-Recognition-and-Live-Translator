import streamlit as st
import csv
import copy
import cv2 as cv
import mediapipe as mp
import av
from model import KeyPointClassifier
from app_files import calc_landmark_list, draw_info_text, draw_landmarks, get_args, pre_process_landmark

from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode



st.title("My first Streamlit app")
st.write("Hello, world")


class VideoProcessor:
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        
        args = get_args()
        
        use_static_image_mode = args.use_static_image_mode
        min_detection_confidence = args.min_detection_confidence
        min_tracking_confidence = args.min_tracking_confidence
        
        mp_hands = mp.solutions.hands
        
        hands = mp_hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        keypoint_classifier = KeyPointClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
            keypoint_classifier_labels = csv.reader(f)
            keypoint_classifier_labels = [
                row[0] for row in keypoint_classifier_labels
            ]
        
        
        image = cv.flip(image, 1)
        
        debug_image = copy.deepcopy(image)
        
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        image.flags.writeable = False
        
        results = hands.process(image)
        
        image.flags.writeable = True
        
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                debug_image = draw_landmarks(debug_image, landmark_list)

                debug_image = draw_info_text(
                    debug_image,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id])
                print(keypoint_classifier_labels[hand_sign_id])

        return av.VideoFrame.from_ndarray(debug_image, format="bgr24")


webrtc_streamer(key="example", video_processor_factory=VideoProcessor)