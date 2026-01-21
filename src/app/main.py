from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.endpoints.monitoring import router as monitoring_router
from app.api.endpoints.train import router as training_router

@asynccontextmanager
async def lifespan(app: FastAPI):

    app.state.trainer = None
    yield
    if hasattr(app.state, "trainer") and app.state.trainer is not None:
        app.state.trainer.cleanup()

app = FastAPI(lifespan=lifespan)

app.include_router(monitoring_router)
app.include_router(training_router)