import os
import io
import loguru
import uvicorn
import asyncio
import aiofiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dookie import main as generate_audio


logger = loguru.logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "Welcome to Sonic Meow!"}


class Prompt(BaseModel):
    ip_address: str


@app.post("/generate")
async def generate(request: Request):
    ip = request.client.host
    if request.host == "":
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    return await waiting_for_file(ip)


async def waiting_for_file(file_path):
    while not os.path.exists(file_path):
        await asyncio.sleep(1)

    count = len(os.listdir("static"))

    if count is None:
        return HTTPException(status_code=500, detail="Error in main function")

    try:
        async with aiofiles.open(file_path, mode="rb") as f:
            content = await f.read()
            byte_array = io.BytesIO(content)
            return JSONResponse(content=byte_array.getvalue(), status_code=200)
    except SystemError as error:
        raise Exception("Error in main function") from error


app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)