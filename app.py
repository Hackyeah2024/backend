from flask import Flask, request, jsonify
import os

from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from moviepy.editor import VideoFileClip
# import speech_recognition as sr

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from nltk import sent_tokenize
import nltk
nltk.download('punkt')

from pydantic import BaseModel, Field
from typing import List
import whisper


from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
openai_api_key = os.environ.get('OPENAI_API_KEY')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def find_off_topic_sentences(main_subject, sentences):
    main_embedding = get_embeddings(main_subject)
    off_topic_sentences = []

    for sentence in sentences:
        sentence_embedding = get_embeddings(sentence)
        similarity = cosine_similarity(
            [main_embedding],
            [sentence_embedding]
        )[0][0]
        if similarity < 0.7:  # Threshold can be adjusted
            off_topic_sentences.append({'text': sentence, 'similarity': similarity})

    return off_topic_sentences

def detect_off_topic_using_embeddings(transcription):
    main_subject = extract_main_subject(transcription)
    sentences = segment_transcript(transcription)
    off_topic_sentences_data = find_off_topic_sentences(main_subject, sentences)

    off_topic_segments = []
    for item in off_topic_sentences_data:
        text = item['text']
        similarity = item['similarity']
        reason = f"Similarity score {similarity:.2f} is below the threshold, indicating the sentence may not be related to the main subject."
        segment = OffTopicSegment(text=text, reason=reason)
        off_topic_segments.append(segment)

    return main_subject, off_topic_segments


def get_embeddings(text):
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
    embedding = embeddings_model.embed_query(text)
    return embedding


def extract_main_subject(transcription):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_api_key)
    prompt_template = PromptTemplate(
        input_variables=["transcription"],
        template="""
Analyze the following speech transcript and provide a concise summary of the main subject or topics introduced at the beginning:

\"\"\"
{transcription}
\"\"\"

Provide the main subject in one or two sentences.
"""
    )
    chain = prompt_template | llm

    main_subject = chain.invoke({"transcription": transcription})
    return main_subject.content


def segment_transcript(transcription):
    sentences = sent_tokenize(transcription)
    return sentences


def transcribe_audio(audio_path):
    try:
        # Load the Whisper model
        model = whisper.load_model("base")

        # Transcribe the audio
        result = model.transcribe(audio_path)

        # Get the transcription text
        transcription = result['text']

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        transcription = "An error occurred during transcription."

    return transcription


# Define the Pydantic models for the JSON schema
class OffTopicSegment(BaseModel):
    text: str = Field(..., description="The off-topic or nonsensical segment")
    reason: str = Field(..., description="Explanation why it does not fit the main subject")


class QualityMetric(BaseModel):
    score: int = Field(..., description="Score from 0 to 10")
    justification: str = Field(..., description="Justification for the score")


class Sentiment(BaseModel):
    overall: str = Field(..., description="Overall sentiment (Positive/Negative/Neutral)")
    emotions_detected: List[str] = Field(..., description="List of emotions detected")


class QualityMetrics(BaseModel):
    clarity_coherence: QualityMetric
    grammar_syntax: QualityMetric
    relevance_to_subject: QualityMetric
    vocabulary_richness: QualityMetric
    sentiment: Sentiment
    filler_words_usage: QualityMetric
    structure_organization: QualityMetric
    llm_off_topic_segments: List[OffTopicSegment]
    persuasiveness: QualityMetric
    key_topics: List[str]


class AnalysisResult(BaseModel):
    main_subject: str
    off_topic_segments: List[OffTopicSegment]
    quality_metrics: QualityMetrics


def analyze_transcription(transcription):

    llm = ChatOpenAI(model="gpt-4o", temperature=0.5, api_key=openai_api_key)

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
    chain = prompt_template | llm | parser


    response = chain.invoke({"transcription": transcription})

    # try:
    #     analysis_dict = response.dict()
    # except Exception as e:
    #     analysis_dict = {"error": f"Failed to parse LLM response: {str(e)}"}
    return response


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
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_path)

    transcription = transcribe_audio(audio_path)

    main_subject, off_topic_segments = detect_off_topic_using_embeddings(transcription)
    quality_metrics = analyze_transcription(transcription)

    analysis = AnalysisResult(
        main_subject=main_subject,
        off_topic_segments=off_topic_segments,
        quality_metrics=quality_metrics
    )

    return jsonify({
        'transcription': transcription,
        'analysis': analysis.dict()
    })


def main():
    app.run(debug=True)


if __name__ == '__main__':
    main()