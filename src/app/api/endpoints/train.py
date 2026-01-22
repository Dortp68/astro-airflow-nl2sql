from fastapi import APIRouter, Request, HTTPException
import threading
from train import (BaseTrainer,
                   UnslothInstructionTrainer,
                   UnslothPreferenceTrainer)
from utils.config import SFTTrainerConfig, SimPOTrainerConfig

router = APIRouter(prefix="/train", tags=["training"])

@router.post("/sft")
def train_sft(request: Request,
    config: SFTTrainerConfig
):
    request.app.state.trainer = UnslothInstructionTrainer(config)

    threading.Thread(
        target=request.app.state.trainer.train,
        daemon=True,
    ).start()

    return {"status": "started"}

@router.post("/stop")
def stop_training(request: Request):
    if hasattr(request.app.state, "trainer") and request.app.state.trainer is not None:
        request.app.state.trainer.stop()
        request.app.state.trainer.cleanup()
    else:
        raise HTTPException(404, "Run not found")

    return {"status": "stopping"}

@router.post("/simpo")
def train_simpo(request: Request,
    config: SimPOTrainerConfig,
):
    request.app.state.trainer = UnslothPreferenceTrainer(config)

    threading.Thread(
        target=request.app.state.trainer.train,
        daemon=True,
    ).start()

    return {"status": "started"}