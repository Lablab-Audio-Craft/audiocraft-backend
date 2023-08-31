import os
import io
from io import BytesIO
import loguru
import uvicorn
import asyncio
import aiofiles
from fastapi import FastAPI, HTTPException, Request
from fastapi.requests import HTTPConnection
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dookie import main as run
import json

logger = loguru.logger

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class formData(BaseModel):
    audio: str
    bpm: int
    iterations: int
    min_dur: Optional[int]
    max_dur: Optional[int]
    dur: Optional[int]


@app.get("/")
def read_root():
    return "localhost:8000/docs"


@app.post("/generate")
async def generate(request: formData):
    form_data: formData = request
    if form_data.audio == "":
        raise HTTPException(status_code=400, detail="audio cannot be empty")
    if form_data.bpm == "":
        raise HTTPException(status_code=400, detail="bpm cannot be empty")
    if form_data.iterations == "":
        raise HTTPException(status_code=400, detail="iterations cannot be empty")
    return await waiting_for_file(form_data=form_data)


async def waiting_for_file(form_data):
    audio, bpm, iterations, min_dur, max_dur, dur = form_data
    file_path = run(audio, bpm, iterations, min_dur, max_dur, dur)
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
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)
