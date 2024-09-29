from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from llm_models import open_ai_llm_mini
from transcript_analysis_models import Summary


def write_summary(text):
    parser = PydanticOutputParser(pydantic_object=Summary)

    prompt_template = PromptTemplate(template="""
            Summarize following text:
            {text}
            Format:\n{format_instructions}
        """,
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()})

    chain = prompt_template | open_ai_llm_mini | parser
    response = chain.invoke({"text": text})

    return response
