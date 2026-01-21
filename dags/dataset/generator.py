from dags.utils import PostgresqlConnector
from dags.utils.prompts import (
    SYNTHETIC_SQL_GENERATION,
    SYNTHETIC_QUERY_GENERATION,
    COMPLEXITY_LEVELS,
)
from dags.dataset.models import (
    SQLSample,
    NLSample,
    InstructionDataset,
    PreferenceDataset,
)
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from typing import List
import logging
from dotenv import load_dotenv
import os


class BaseGenerator(ABC):
    @abstractmethod
    def generate_dataset(self, num_samples: int):
        pass


def normalize_sql(sql: str) -> str:
    return " ".join(sql.lower().split())


class InstructionDatasetGenerator(BaseGenerator):
    def __init__(self):
        load_dotenv()
        base_url = os.getenv("BASE_URL")
        api_key = os.getenv("API_KEY")
        model = os.getenv("CODE_MODEL")
        logging.info(base_url)
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=2000,
            temperature=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_retries=3,
        )
        self.db = PostgresqlConnector()
        self.schema = self.db.get_schema()
        self.dataset = InstructionDataset(
            instruction=SQLSample(queries=[]), answer=NLSample(queries=[])
        )

    def generate_sql(self, complexity: int, batch_size: int) -> List[str]:
        prompt = SYNTHETIC_SQL_GENERATION.format(
            db_schema=self.schema,
            num_questions=batch_size,
            complexity=complexity,
            description=COMPLEXITY_LEVELS[complexity],
        )
        model_structured = self.model.with_structured_output(SQLSample)
        response = model_structured.invoke(prompt)
        return response

    def generate_nl(self, query: str, num_questions: int) -> List[str]:
        prompt = SYNTHETIC_QUERY_GENERATION.format(
            db_schema=self.schema, sql_query=query, num_questions=num_questions
        )
        model_structured = self.model.with_structured_output(NLSample)
        response = model_structured.invoke(prompt)
        return response

    def validate_sql(self, queries: List[str]) -> List[str]:
        unique_sql = list({normalize_sql(q): q for q in queries}.values())
        unique_sql = [q for q in unique_sql if self.db.validate_query(q)]
        logging.info(f"Number of queries after validation: {len(unique_sql)}")
        return unique_sql

    def generate_dataset(self, num_samples: int):
        batch_size = num_samples // len(COMPLEXITY_LEVELS)
        for complexity in COMPLEXITY_LEVELS:
            sql_queries = self.generate_sql(
                complexity=complexity, batch_size=batch_size
            )
            validated_sql = self.validate_sql(queries=sql_queries)

            for sql in validated_sql:
                nl_queries = self.generate_nl(query=sql, num_questions=1)
                for nl in nl_queries:
                    self.dataset.instructions.append(nl)
                    self.dataset.answers.append(sql)

    def generate_dataset_simple(self):
        pass


class PreferenceDatasetGenerator(BaseGenerator):
    def __init__(self, instruction_dataset: InstructionDataset):
        load_dotenv()
        base_url = os.getenv("OPENROUTER_BASE_URL")
        api_key = os.getenv("OPENROUTER_API_KEY")
        model = os.getenv("WEAK_MODEL")
        logging.info(base_url)
        self.model = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_tokens=2000,
            temperature=1.0,
            frequency_penalty=0.5,
            presence_penalty=0.5,
            max_retries=3,
        )
        self.db = PostgresqlConnector()
        self.schema = self.db.get_schema()
        self.dataset = PreferenceDataset()
        self.instruction_dataset = instruction_dataset

    def generate_rejected_sql(self, question: str) -> str:
        prompt = f"""Generate a SQL query for the following natural language question based on the database schema.

Database Schema:
{self.schema}

Question: {question}

Return only the SQL query, no explanations."""
        response = self.model.invoke(prompt)
        return response.content.strip()

    def generate_dataset(self, num_samples: int):
        if instruction_dataset is None:
            raise ValueError(
                "Instruction dataset is required for PreferenceDatasetGenerator"
            )
        num_samples = min(num_samples, len(instruction_dataset.instructions))

        for i in range(num_samples):
            instr = instruction_dataset.instructions[i]
            chosen = instruction_dataset.answers[i]
            nl_question = instr["queries"][0] if isinstance(instr, dict) else str(instr)
            rejected = self.generate_rejected_sql(nl_question)
            self.dataset.instructions.append(instr)
            self.dataset.answers.append(chosen)
            self.dataset.rejected.append(rejected)
