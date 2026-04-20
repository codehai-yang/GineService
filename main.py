import uvicorn
from fastapi import FastAPI
from app.api.predict_controller import router

app = FastAPI()
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        workers=1,
        log_level="info"
    )