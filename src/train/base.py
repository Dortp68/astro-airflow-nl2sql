from abc import ABC, abstractmethod
from utils.config import BaseTrainerConfig
import threading
import torch
import gc

class BaseTrainer(ABC):
    def __init__(self, config: BaseTrainerConfig):
        self.config = config
        self.stop_event = threading.Event()
        self.model = None
        self.tokenizer = None
        self.trainer = None

    def stop(self):
        """Signal training interruption."""
        self.stop_event.set()

    def is_stopped(self) -> bool:
        return self.stop_event.is_set()

    @abstractmethod
    def setup_model(self):
        """Load and configure the model."""
        pass

    @abstractmethod
    def load_and_preprocess_data(self):
        """Load and preprocess the dataset."""
        pass

    @abstractmethod
    def setup_trainer(self, dataset):
        """Set up the trainer."""
        pass

    @abstractmethod
    def save_model(self):
        """Save the trained model."""
        pass

    @abstractmethod
    def train(self):
        """Run the training process."""
        pass

    @staticmethod
    def check_cuda():
        """Check if the cuda is available."""
        return torch.cuda.is_available()

    def cleanup(self):
        """Cleaning up GPU and Python memory"""
        if self.trainer:
            del self.trainer
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        torch.cuda.empty_cache()

