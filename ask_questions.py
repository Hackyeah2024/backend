from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from llm_models import open_ai_llm_mini
from transcript_analysis_models import Questions


def ask_questions(text):
    parser = PydanticOutputParser(pydantic_object=Questions)

    prompt_template = PromptTemplate(template="""
            Ask 10 questions in Polish language for following text:
            {text}
            Format:\n{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()})

    chain = prompt_template | open_ai_llm_mini | parser
    response = chain.invoke({"text": text})

    return response
