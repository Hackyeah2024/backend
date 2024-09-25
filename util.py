from langchain_core.prompts import PromptTemplate
from nltk import sent_tokenize

from llm_models import open_ai_llm


def segment_transcript(transcription):
    sentences = sent_tokenize(transcription)
    return sentences


def extract_main_subject(transcription):
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
    chain = prompt_template | open_ai_llm

    main_subject = chain.invoke({"transcription": transcription})
    return main_subject.content
