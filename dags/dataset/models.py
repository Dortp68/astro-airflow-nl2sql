from pydantic import BaseModel, Field
from typing import List, TypedDict

class SQLSample(TypedDict):
    queries: List[str]

class NLSample(TypedDict):
    queries: List[str]

class InstructionDataset(BaseModel):
    instructions: SQLSample = Field(default=[], description="Instruction")
    answers: NLSample = Field(default=[], description="Answer sql query")

class PreferenceDataset(BaseModel):
    instructions: SQLSample = Field(default=[], description="Instruction")
    answers: NLSample = Field(default=[], description="Answer sql query")
    rejected: List[str] = Field(default=[], description="Rejected Answer sql query")