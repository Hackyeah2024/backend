from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from llm_models import open_ai_llm_mini
from transcript_analisis_models import SubtitlesAnalysis


def compare_subtitles(transcription, ocr):
    ocr_subtitles = " ".join(sentence["text"] for sentence in ocr)

    parser = PydanticOutputParser(pydantic_object=SubtitlesAnalysis)

    prompt_template = PromptTemplate(template="""
            Rate similarity between two versions of subtitles for video between 0% and 100%.
            Ignore minor changes like mismatch between single letters.
            Transcription subtitles:\n{transcription_subtitles}\n
            OCR subtitles:\n{ocr_subtitles}\n
            Format:\n{format_instructions}
        """,
        input_variables=["transcription_subtitles", "ocr_subtitles"],
        partial_variables={"format_instructions": parser.get_format_instructions()})

    chain = prompt_template | open_ai_llm_mini | parser
    response = chain.invoke({
        "transcription_subtitles": transcription,
        "ocr_subtitles": ocr_subtitles})

    return response
