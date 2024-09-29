import io
from contextlib import redirect_stdout

from google.cloud import videointelligence



def detect_subtitles(path):
    # Initialize the Video Intelligence client
    video_client = videointelligence.VideoIntelligenceServiceClient()

    # Define the features to detect text and persons in the video
    features = [
        videointelligence.Feature.TEXT_DETECTION,
        videointelligence.Feature.PERSON_DETECTION
    ]

    # Configure person detection settings
    person_detection_config = videointelligence.PersonDetectionConfig(
        include_bounding_boxes=True,
        include_pose_landmarks=False,
        include_attributes=True
    )

    # Set the video context with the person detection configuration
    video_context = videointelligence.VideoContext(
        person_detection_config=person_detection_config
    )

    # Read the video file
    with io.open(path, "rb") as file:
        input_content = file.read()

    # Create the video annotation request with both features
    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": video_context,
        }
    )

    print("Processing video for text and person detection.")
    result = operation.result(timeout=600)

    # Process the text detection annotations
    annotation_result = result.annotation_results[0]
    subtitles = []

    for text_annotation in annotation_result.text_annotations:
        vertices = text_annotation.segments[0].frames[0].rotated_bounding_box.vertices
        if vertices[0].x < 0.5 and vertices[0].y > 0.8 and \
           vertices[1].x > 0.5 and vertices[1].y > 0.8 and \
           vertices[2].x > 0.5 and vertices[2].y < 1 and \
           vertices[3].x < 0.5 and vertices[3].y < 1:

            text = text_annotation.text.split(">")[-1].strip()
            start_time = text_annotation.segments[0].segment.start_time_offset
            end_time = text_annotation.segments[-1].segment.end_time_offset

            if text and not any(s["text"] == text for s in subtitles):
                subtitles.append({
                    "text": text,
                    "start_time": start_time.seconds + start_time.microseconds * 1e-6,
                    "end_time": end_time.seconds + end_time.microseconds * 1e-6,
                    "confidence": text_annotation.segments[0].confidence,
                    "text_box": "({x1}, {y1}) -> ({x2}, {y2})".format(x1=vertices[0].x, y1=vertices[0].y, x2=vertices[3].x, y2=vertices[3].y)
                })

    # Sort the detected subtitles
    sorted_subtitles = sorted(sorted(subtitles, key=lambda d: d["text_box"].split(",")[1]), key=lambda d: d['start_time'])

    # # Process the person detection annotations
    bounding_boxes = []
    with open('out.txt', 'w') as f:
        with redirect_stdout(f):
            print(annotation_result.person_detection_annotations)
    for person_annotation in annotation_result.person_detection_annotations:
        for track in person_annotation.tracks:
            for timestamped_object in track.timestamped_objects:
                bounding_box = {
                    "time_offset": timestamped_object.time_offset.seconds + timestamped_object.time_offset.microseconds * 1e-6,
                    "left": timestamped_object.normalized_bounding_box.left,
                    "top": timestamped_object.normalized_bounding_box.top,
                    "right": timestamped_object.normalized_bounding_box.right,
                    "bottom": timestamped_object.normalized_bounding_box.bottom,
                }
                bounding_boxes.append(bounding_box)


    # Return the results of both text detection and person detection
    return sorted_subtitles, bounding_boxes
