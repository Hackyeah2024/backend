from typing import List, Optional
from enum import Enum

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from llm_models import open_ai_llm_mini, open_ai_llm, command_r_plus_llm


class FactStatus(str, Enum):
    MOSTLY_TRUE = "MOSTLY_TRUE"
    VERIFIED = "VERIFIED"
    FALSE = "FALSE"
    CONSPIRACY_THEORY = "CONSPIRACY_THEORY"
    COMPLETE_NONSENSE = "COMPLETE_NONSENSE"
    LIES_AND_MISTIFICATION = "LIES_AND_MISTIFICATION"
    COULD_NOT_VERIFY = "COULD_NOT_VERIFY"


class FactCheckResultDetails(BaseModel):
    status: FactStatus = Field(None, description="Fact given to verify")
    explanation: Optional[str] = Field(None, description="Explanation of resulting status")
    name_of_knowledge_source: Optional[List[str]] = Field(None, description="List of names of source of this information")

class FactCheck(BaseModel):
    fact: Optional[str] = Field(None, description="Fact given to verify")
    details: Optional[FactCheckResultDetails] = Field(None, description="Thorough details of fact verification")

class FactCheckResult(BaseModel):
    data: List[FactCheck]


# Function to analyze a single segment independently
def verify_facts(facts: List[str]) -> List[FactCheck]:
    parser = PydanticOutputParser(pydantic_object=FactCheck)

    prompt_template = PromptTemplate(template="""
        You are an expert fact verification analyst. Verify in internet fact provided in context. Return json output without any wrapping like ```json ```

        1. Verify in internet all passed facts.
        2. Give it proper status with explanation why exact status was assigned.
        3. Provide all sources used for verifying fact

        Format:
        {format_instructions}
            
        Facts to verify list:
        \"\"\"
        {fact}
        \"\"\"
    """,

     input_variables=["fact"],
     partial_variables={"format_instructions": parser.get_format_instructions()},
   )

    chain = prompt_template | command_r_plus_llm | parser


    facts_checks = []
    for index, f in enumerate(facts):
        response = chain.invoke({"fact": f})
        facts_checks.append(response.dict())


    return facts_checks

