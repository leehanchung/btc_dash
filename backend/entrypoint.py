import uvicorn
from app import app


if __name__ == "__main__":
    uvicorn.run("app:app", log_config=None, host="0.0.0.0", port=5000, reload=True)
