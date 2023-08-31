import json
from text.test_openai import run_text_generation

from fastapi import HTTPException


class DataHandler:
    def __init__(self):
        self.run_text_generation = run_text_generation

    def handle_command_chat(self, content: str, role: str = None) -> str:
        role = role
        if not role:
            role = "user"
        if not role in ["user", "assistant", "system"]:
            role = "user"
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Content is required. Content is a string of text meant to be sent to the chat bot api.",
            )

        assistant_message = (
            self.text.send_chat_complete(messages=messages).choices[0].message
        )
        return assistant_message.content

    def handle_chat(self, content: str, role: str = None) -> str:
        role = role
        if not role:
            role = "user"
        if not role in ["user", "assistant", "system"]:
            role = "user"
        if not content:
            raise HTTPException(
                status_code=400,
                detail="Content is required. Content is a string of text meant to be sent to the chat bot api.",
            )
        message = self.messages.create_message(role=role, content=content)
        messages = []
        self.context.add_message(message=message)
        for message in self.context.get_context():
            messages.append({"content": message.content, "role": message.role})
        assistant_message = (
            self.text.send_chat_complete(messages=messages).choices[0].message
        )
        self.context.add_message(message=assistant_message)
        return assistant_message.content

    def handle_image(self, prompt: str) -> str:
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="Prompt is required. Prompt is a string of text meant to be sent to the image api.",
            )
        return self.image.generate_image(prompt=prompt)


if __name__ == "__main__":
    DataHandler()
