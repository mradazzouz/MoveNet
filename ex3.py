import cv2
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer
import av

# Set CORS headers to allow same-origin
st.set_page_config(
    page_title="My Streamlit App",
    page_icon=":sunglasses:"
)

# Load model
interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()

# Draw keypoints
def draw_keypoints(frame, keypoints, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

# Draw edges
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def draw_edges(frame, keypoints, edges, confidence):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence) & (c2 > confidence):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)


# Create a VideoProcessor class
class VideoProcessor:
    def __init__(self) -> None:
        self.confidence_threshold = 0.6

    def recv(self, frame):
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2RGB)

        # Resize and preprocess the frame for the model
        img = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        # Setup input and output details for the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])


        # Draw edges and keypoints on the frame
        draw_edges(frame, keypoints_with_scores, EDGES, self.confidence_threshold)
        draw_keypoints(frame, keypoints_with_scores, self.confidence_threshold)

        # Convert the frame back to BGR format
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return av.VideoFrame.from_ndarray(frame, format="bgr24")

# Create WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# Add sliders to adjust confidence threshold
if webrtc_ctx.video_processor:
    webrtc_ctx.video_processor.confidence_threshold = st.slider(
        "Confidence Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.6
    )


