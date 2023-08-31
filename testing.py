import base64
import requests
import loguru
import json
import io

logger = loguru.logger


def main():
    with open("in/Off_Kilter_[Master]-Bako-48k-32Bit-1db-1.mp3", "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")

    data = {
        "bpm": 0,
        "iterations": 0,
        "min_dur": 0,
        "max_dur": 0,
        "dur": 0,
    }

    payload = {**data, "audio": file_content}

    url = "http://localhost:8000/generate"

    response = requests.post(url, data=payload)

    logger.info(response.status_code)
    print(response.json())


if __name__ == "__main__":
    main()
