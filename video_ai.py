import io
from google.cloud import videointelligence

def video_detect_text(path):
    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.TEXT_DETECTION]
    video_context = videointelligence.VideoContext()

    with io.open(path, "rb") as file:
        input_content = file.read()

    operation = video_client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": video_context,
        }
    )

    print("Processing video for text detection.")
    result = operation.result(timeout=300)

    annotation_result = result.annotation_results[0]
    subtitles = []

    for text_annotation in annotation_result.text_annotations:
        vertices = text_annotation.segments[0].frames[0].rotated_bounding_box.vertices
        if vertices[0].x > 0 and vertices[0].y > 0.8 and \
            vertices[1].x < 0.8 and vertices[1].y > 0.8 and \
            vertices[2].x < 0.8 and vertices[2].y < 0.99 and \
            vertices[3].x > 0 and vertices[3].y < 0.99:

            text = text_annotation.text.replace("< MF K$ C< CIRF >", "").strip()
            start_time = text_annotation.segments[0].segment.start_time_offset
            end_time = text_annotation.segments[len(text_annotation.segments) - 1].segment.end_time_offset

            if text and not text == any(subtitles["text"]):
                subtitles.append({
                    "text": text,
                    "start_time": start_time.seconds + start_time.microseconds * 1e-6,
                    "end_time": end_time.seconds + end_time.microseconds * 1e-6,
                    "confidence": text_annotation.segments[0].confidence
                })
    print(subtitles)