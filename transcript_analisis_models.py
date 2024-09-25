from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from llm_models import open_ai_llm_mini


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
