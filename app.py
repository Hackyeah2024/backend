from flask import Flask, request, jsonify
import os

# import speech_recognition as sr

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import nltk

from audio import transcribe, extract_audio_file
from llm_models import open_ai_llm
from offtopic import detect_off_topic_using_embeddings
from transcript_analisis_models import OffTopicSegment, QualityMetrics, analyze_segment, analyze_segments_comparatively, \
    EventAnalysis

nltk.download('punkt')

from pydantic import BaseModel
from typing import List

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
openai_api_key = os.environ.get('OPENAI_API_KEY')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def analyze_transcription(transcription):

    parser = PydanticOutputParser(pydantic_object=QualityMetrics)

    prompt_template = PromptTemplate(
        input_variables=["transcription"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
            You are an expert speech analyst. Analyze the following transcribed speech and provide the analysis.
            1. Clarity and Coherence (score out of 10):
               - Justification:
            
            2. Grammar and Syntax (score out of 10):
               - Justification:
            
            3. Relevance to Main Subject (score out of 10):
               - Justification:
            
            4. Vocabulary Richness (score out of 10):
               - Justification:
            
            5. Sentiment Analysis:
               - Overall sentiment (Positive/Negative/Neutral):
               - Emotions detected:
            
            6. Use of Filler Words (score out of 10):
               - List of filler words used:
            
            7. Structure and Organization (score out of 10):
               - Justification:
            
            8. Persuasiveness (score out of 10):
               - Justification:
            
            9. Key Topics Discussed:
               - List of topics:
            
            Format:
            {format_instructions}
            
            Transcription:
            \"\"\"
            {transcription}
            \"\"\"
        """
    )
    chain = prompt_template | open_ai_llm | parser


    response = chain.invoke({"transcription": transcription})

    return response


class AnalysisResult(BaseModel):
    main_subject: str
    off_topic_segments: List[OffTopicSegment]
    quality_metrics: QualityMetrics


@app.route('/process_video', methods=['POST'])
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
    quality_metrics = analyze_transcription(transcription)

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

    return jsonify({
        'transcription': transcription,
        'analysis': analysis.dict(),
        "segments_analysis": [segment_analysis.dict() for segment_analysis in segments_analysis],
        "events": [event.dict() for event in events]
    })


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()