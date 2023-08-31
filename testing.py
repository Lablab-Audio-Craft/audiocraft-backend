import base64
import requests
import loguru
import os

logger = loguru.logger


def main():
    # Read the file and base64 encode it
    audio = f"static/{os.listdir('static/')[0]}"
    with open(audio, "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")

    # Other form data
    detail = {"audio": file_content}

    # Endpoint URL
    url = "http://localhost:8001/generate"

    # Make the request
    response = requests.post(url, data=detail, timeout=6000)

    # Log and print the response
    logger.info(response.status_code)
    print(response)


if __name__ == "__main__":
    main()
