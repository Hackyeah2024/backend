from typing import List, Optional
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from llm_models import open_ai_llm_mini, open_ai_llm


class SentimentType(str, Enum):
    VERY_NEGATIVE = "VERY_NEGATIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
    VERY_POSITIVE = "VERY_POSITIVE"

class OffTopicSegment(BaseModel):
    text: str = Field(..., description="The off-topic or nonsensical segment")
    reason: str = Field(..., description="Explanation why it does not fit the main subject")
    segment_index: int = Field(..., description="Index of a segment that is associated with off-topic")


class QualityMetric(BaseModel):
    score: int = Field(..., description="Score from 0 to 10")
    justification: str = Field(..., description="Justification for the score")


class Sentiment(BaseModel):
    overall: SentimentType = Field(..., description="Overall sentiment ")
    emotions_detected: List[str] = Field(..., description="List of emotions detected")

    class Config:
        use_enum_values = True


class SegmentsCategorization(BaseModel):
    category: str = Field(..., description="Category for consecutive segments.")
    from_segment: int = Field(..., description="Starting index of segment where this category matches")
    to_segments: int = Field(..., description="Ending index of segment where this category matches")

class FactDetail(BaseModel):
    fact: str = Field(..., description="A short version of a fact")
    fact_with_more_context: str = Field(..., description="A fact that is self contained with a context and all details, so that later on it can be verified by other llm with internet access")


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
    categorized_segments: List[SegmentsCategorization]

    facts_to_verify: List[FactDetail] = Field(..., description="All information presented as facts that user ought to carefully verify with own research")




class SegmentAnalysis(BaseModel):
    clarity_coherence: int = Field(..., description="Score out of 10")
    sentiment: str = Field(..., description="Sentiment: Positive/Negative/Neutral")
    key_topics: List[str] = Field(..., description="List of key topics discussed")

    class Config:
        schema_extra = {
            "example": {
                "clarity_coherence": 8,
                "sentiment": "Neutral",
                "key_topics": ["topic1", "topic2"]
            }
        }


class ComparativeAnalysis(BaseModel):
    changes_in_sentiment: Optional[str] = Field(None, description="Description of any changes in sentiment")
    changes_in_topics: Optional[List[str]] = Field(None, description="List of topics changed")
    significant_events: Optional[str] = Field(None, description="Description of any significant events or shifts")

    class Config:
        schema_extra = {
            "example": {
                "changes_in_sentiment": "Shifted from Neutral to Positive",
                "changes_in_topics": ["New topic introduced"],
                "significant_events": "Speaker switched from discussing Topic A to Topic B."
            }
        }

class EventAnalysis(BaseModel):
    index: int
    from_segment: int
    to_segment: int
    event_analysis: ComparativeAnalysis


# Function to analyze a single segment independently
def analyze_segment(segment_transcription: str) -> SegmentAnalysis:
    parser = PydanticOutputParser(pydantic_object=SegmentAnalysis)

    prompt_template = PromptTemplate( template = """
        You are an expert speech analyst. Analyze the following transcribed speech segment and provide the analysis.

        1. Clarity and Coherence (score out of 10):
        2. Sentiment Analysis (Positive/Negative/Neutral):
        3. Key Topics Discussed (List of topics):
        4. Always return data in language of a transcript.
       Format:
        {format_instructions}
            
        Transcription:
        \"\"\"
        {segment_transcription}
        \"\"\"
    """,

      input_variables=["segment_transcription"],
      partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt_template | open_ai_llm_mini | parser
    response = chain.invoke({"segment_transcription": segment_transcription})

    return response


# Function to perform comparative analysis between two consecutive segments
def analyze_segments_comparatively(previous_segment: dict[str, any],
                                   current_segment: dict[str, any]) -> ComparativeAnalysis:

    parser = PydanticOutputParser(pydantic_object=ComparativeAnalysis)

    prompt_template = PromptTemplate( template ="""
        You are an expert speech analyst. Compare the following two transcribed speech segments and provide the analysis.
        Always return data in language of a transcript.
        
        Segment 1:
        \"\"\"
        {previous_text}
        \"\"\"

        Segment 2:
        \"\"\"
        {current_text}
        \"\"\"

       Format:
        {format_instructions}
            
        Compare and analyze:
        1. Changes in Sentiment:
        2. Changes in Topics Discussed:
        3. Any Significant Events or Shifts:

        Provide details for any detected changes or events.
    """,

      input_variables=["previous_text", "current_text"],
      partial_variables={"format_instructions": parser.get_format_instructions()},

    )
    chain = prompt_template | open_ai_llm_mini | parser
    response = chain.invoke({
        "previous_text": previous_segment["text"],
        "current_text": current_segment["text"]
    })

    return response


def analyze_transcription(segments: List[dict[str, any]]):

    parser = PydanticOutputParser(pydantic_object=QualityMetrics)

    prompt_template = PromptTemplate(
        input_variables=["transcription"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
        template="""
            You are an expert speech analyst. Analyze the following transcribed speech and provide the analysis.
            You receive index segments so that you can detect certain aspects of transcript for consecutive segments.
            Always return data in language of a transcript.
            
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
            
            10. Grouped segments categorization:
               - List of grouped segments by category.:
               - Group segments together into logically coherent clusters.
            11. All interesting informations presented as facts that a viewer should verifi:
               - List all important information presented as facts that can be verified by research:

               
            Format:
            {format_instructions}
            
            Transcription:
            \"\"\"
            {transcription}
            \"\"\"
        """
    )
    chain = prompt_template | open_ai_llm | parser

    transcription = [f"{index}: {d['text']}" for index, d in enumerate(segments)]

    response = chain.invoke({"transcription": transcription})

    return response


class AnalysisResult(BaseModel):
    main_subject: str
    off_topic_segments: List[OffTopicSegment]
    quality_metrics: QualityMetrics
