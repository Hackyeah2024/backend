import os

from flask import request, jsonify
from flask_cors import cross_origin

from controller.core import app
from audio import extract_audio_file, transcribe
from offtopic import detect_off_topic_using_embeddings
from transcript_analisis_models import analyze_transcription, analyze_segments_comparatively, EventAnalysis, \
    analyze_segment, AnalysisResult
from video_ai import detect_subtitles
from compare_subtitles import compare_subtitles


@app.route('/', methods=['GET'])
def health_check():
    # You can add any logic here to verify your app's health
    return jsonify(status="healthy"), 200

@app.route('/process_video', methods=['POST'])
@cross_origin()
def process_video():
    if 'video_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['video_file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(video_path)

    audio_path = os.path.splitext(video_path)[0] + '.wav'
    extract_audio_file(audio_path, video_path)

    transcription,segments = transcribe(audio_path)

    main_subject, off_topic_segments = detect_off_topic_using_embeddings(transcription)
    quality_metrics = analyze_transcription(segments)

    # Perform comparative analysis for each pair of consecutive segments
    events = []
    for i in range(1, len(segments)):
        event_analysis = analyze_segments_comparatively(segments[i - 1], segments[i])
        event_record = EventAnalysis(
            index=i,
            from_segment=i - 1,
            to_segment=i,
            event_analysis=event_analysis
        )
        events.append(event_record)
    # Analyze each segment independently
    segments_analysis = []
    for segment in segments:
        analysis = analyze_segment(segment["text"])
        segments_analysis.append(analysis)

    analysis = AnalysisResult(
        main_subject=main_subject,
        off_topic_segments=off_topic_segments,
        quality_metrics=quality_metrics

    )

    detected_subtitles = detect_subtitles(video_path)
    subtitles_matching = compare_subtitles(segments, detected_subtitles)

    return jsonify({
        'transcription': segments,
        'analysis': analysis.dict(),
        "segments_analysis": [segment_analysis.dict() for segment_analysis in segments_analysis],
        "events": [event.dict() for event in events],
        "subtitles_matching": subtitles_matching.dict()
    })
