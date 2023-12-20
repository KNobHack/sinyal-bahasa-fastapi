
import os
import uvicorn
import traceback
import cv2
import numpy as np
import tempfile

from fastapi import FastAPI, Response, UploadFile

from model import extract_keypoints, mediapipe_detection, get_model, mp_holistic, actions


app = FastAPI()

# This endpoint is for a test (or health check) to this server
@app.get("/")
def index():
    return "API is working!"

@app.post("/translate_video")
async def translate_video(video: UploadFile, response: Response):
    try:
        # Checking if it's an image
        if video.content_type not in ["video/mp4"]:
            response.status_code = 400
            return "File is Not a Video!"
        
        # Open the video file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
            temp_video_path = temp_video_file.name
            temp_video_file.write(await video.read())

        cap = cv2.VideoCapture(temp_video_path)

        model = get_model()
        response_text = []
        sequence = []

        # Loop through the frames and process each frame
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            word = ""
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret:
                    break

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if word != actions[np.argmax(res)]:
                        response_text.append(actions[np.argmax(res)])
                        word = actions[np.argmax(res)]

        # Release the VideoWriter and VideoCapture objects
        # out.release()
        # video.release()
        cap.release()
        
        return { "prediction": " ".join(response_text) }
    except Exception as e:
        traceback.print_exc()
        response.status_code = 500
        return "Internal Server Error"


port = os.environ.get("PORT", 8080)
uvicorn.run(app, host='0.0.0.0', port=port)