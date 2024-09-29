import cv2, os, math
from google.cloud import vision


def detect_emotions(video_path):
    client = vision.ImageAnnotatorClient()

    vidcap = cv2.VideoCapture(video_path)
    # frame_rate = vidcap.get(cv2.CAP_PROP_FPS)
    # while vidcap.isOpened():
    #     frame_id = vidcap.get(1)  # current frame number
    #     ret, frame = vidcap.read()
    #     if not ret:
    #         break

        # if frame_id % math.floor(frame_rate) == 0:
        #     print("Processing frame {frame_id}".format(frame_id=frame_id))
        #     image = vision.Image(content=frame)
        #     response = client.face_detection(image=image)
        #     faces = response.face_annotations
        #     likelihood_name = (
        #         "UNKNOWN",
        #         "VERY_UNLIKELY",
        #         "UNLIKELY",
        #         "POSSIBLE",
        #         "LIKELY",
        #         "VERY_LIKELY",
        #     )
        #     for face in faces:
        #         print(f"anger: {likelihood_name[face.anger_likelihood]}")
        #         print(f"joy: {likelihood_name[face.joy_likelihood]}")
        #         print(f"surprise: {likelihood_name[face.surprise_likelihood]}")

        #     path = os.path.abspath("frames/frame%d.jpg" % int(frameId))
        #     cv2.imwrite(path, frame)
    duration = vidcap.get(cv2.CAP_PROP_POS_MSEC)
    vidcap.release()
    return None, duration
