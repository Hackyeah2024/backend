from pydantic import BaseModel, ValidationError
import os
from typing import List, Optional

from audio import extract_audio_file, transcribe
from controller.core import app
from flask import Blueprint, request, jsonify

from transcript_analisis_models import analyze_segments_comparatively, analyze_segment, analyze_transcription, \
    FactDetail
from fact_check_models import verify_facts
from util import segment_transcript


class SegmentDTO(BaseModel):
    previous_segment: str
    current_segment: str

@app.route('/test_ai_segment_comparatively', methods=['POST'])
def test_compare_segments_analisis():
    try:
        # Parse and validate incoming request data
        data = request.get_json()
        segment_data = SegmentDTO(**data)

        # Call the service function
        result = analyze_segments_comparatively(
            previous_segment={ "text": segment_data.previous_segment },
            current_segment={ "text": segment_data.current_segment }
        )

        return jsonify(result.dict()), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during analysis.', 'reason': str(e)}), 500

@app.route('/test_ai_segment', methods=['POST'])
def test_segments_analisis():
    try:
        # Parse and validate incoming request data
        data = request.get_json()
        segment_data = SegmentDTO(**data)

        # Call the service function
        result = analyze_segment(
            segment_data.current_segment
        )

        return jsonify(result.dict()), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during analysis.', 'reason': str(e)}), 500




class TranscriptDTO(BaseModel):
    transcript: str

@app.route('/test_ai_transcript', methods=['POST'])
def test_segments_transcript():
    try:
        # Parse and validate incoming request data
        data = request.get_json()
        transcript_data = TranscriptDTO(**data)
        segments = []
        sentences = segment_transcript(transcript_data.transcript)

        for sentence in sentences:
           segments.append({'text': sentence})

        # Call the service function
        result = analyze_transcription(
            segments
        )

        return jsonify(result.dict()), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during analysis.', 'reason': str(e)}), 500

@app.route('/test_transcribe', methods=['POST'])
def test_transcribe():
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

    return jsonify({"segments": segments, "transcription": transcription}), 200



class FactsVerificationDTO(BaseModel):
    facts_to_verify: List[FactDetail]

@app.route('/test_facts_verification', methods=['POST'])
def test_facts_verification():
    try:
        # Parse and validate incoming request data
        data = request.get_json()
        facts_data = FactsVerificationDTO(**data)

        facts = []

        for f in facts_data.facts_to_verify:
           facts.append(f.fact_with_more_context)

        # Call the service function
        result = verify_facts(
            facts
        )

        return jsonify(result), 200

    except ValidationError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'An error occurred during analysis.', 'reason': str(e)}), 500