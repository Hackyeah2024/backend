import io
from google.cloud import videointelligence

def detect_subtitles(path):
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
        if vertices[0].x < 0.5 and vertices[0].y > 0.8 and \
            vertices[1].x > 0.5 and vertices[1].y > 0.8 and \
            vertices[2].x > 0.5 and vertices[2].y < 1 and \
            vertices[3].x < 0.5 and vertices[3].y < 1:

            text = text_annotation.text.split(">")[-1].strip()
            start_time = text_annotation.segments[0].segment.start_time_offset
            end_time = text_annotation.segments[len(text_annotation.segments) - 1].segment.end_time_offset

            if text and not any(s["text"] == text for s in subtitles):
                subtitles.append({
                    "text": text,
                    "start_time": start_time.seconds + start_time.microseconds * 1e-6,
                    "end_time": end_time.seconds + end_time.microseconds * 1e-6,
                    "confidence": text_annotation.segments[0].confidence,
                    "text_box": "({x1}, {y1}) -> ({x2}, {y2})".format(x1=vertices[0].x, y1=vertices[0].y, x2=vertices[3].x, y2=vertices[3].y)
                })

    sorted_subtitles = sorted(sorted(subtitles, key=lambda d: d["text_box"].split(",")[1]), key=lambda d: d['start_time'])
    return sorted_subtitles