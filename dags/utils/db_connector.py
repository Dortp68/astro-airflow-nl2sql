from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
from abc import ABC, abstractmethod
import logging
import os

class BaseDatabaseConnector(ABC):
    @abstractmethod
    def get_schema(self) -> str:
        """Get schema information for database"""
        pass

    @abstractmethod
    def run_query(self, query):
        """Execute a SQL query and return results."""
        pass

    @abstractmethod
    def validate_query(self, query) -> bool:
        """Validate a SQL query"""
        pass

class PostgresqlConnector(BaseDatabaseConnector):
    def __init__(self):
        load_dotenv()
        db_host = os.getenv("DB_HOST")
        db_port = os.getenv("DB_PORT")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")
        uri = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        logging.info(uri)
        self.db = SQLDatabase.from_uri(uri)

    def get_schema(self) -> str:
        """Get schema information for database"""
        return self.db.get_table_info()

    def run_query(self, query):
        """Execute a SQL query and return results."""
        return self.db.run(query)

    def validate_query(self, query) -> bool:
        """Validate a SQL query"""
        try:
            result = self.db.run("EXPLAIN " + query)
            return True
        except Exception as e:
            return False
