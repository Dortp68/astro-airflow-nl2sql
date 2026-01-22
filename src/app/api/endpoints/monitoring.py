from fastapi import APIRouter

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/live")
async def liveness():
    return {"status": "alive"}


@router.get("/ready", include_in_schema=False)
async def readiness():

    return {"status": "ready", "model_loaded": True}


@router.get("/health", include_in_schema=False)
async def health():
    return {"status": "healthy"}

    return Response(status_code=503)


@router.get("/ping")
async def ping():
    return Response(status_code=204)


@router.get("/status")
async def status():
    return {
        "status": "ok"
    }
    # return {
    #     "status": "ok" if model_ready else "initializing",
    #     "model_ready": model_ready,
    #     "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    #     "startup_complete": startup_complete,
    #     # + gpu memory, model version, etc.
    # }