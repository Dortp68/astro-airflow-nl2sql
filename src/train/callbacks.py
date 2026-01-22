from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import threading

class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_event: threading.Event):
        self.stop_event = stop_event

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if self.stop_event.is_set():
            control.should_training_stop = True
            control.should_save = True
            control.should_log = True
            return control