from sklearn.metrics.pairwise import cosine_similarity

from transcript_analisis_models import OffTopicSegment
from util import segment_transcript, extract_main_subject
from embeddings import get_embeddings


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
