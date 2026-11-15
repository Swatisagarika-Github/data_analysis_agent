
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


def generate_ai_response(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key not found. Check your .env file."
        )

    try:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=15000
            )
        )

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if not response.text:
            raise ValueError(
                "Gemini returned an empty response."
            )

        return response.text

    except Exception as error:
        raise RuntimeError(
            f"Gemini API request failed: {error}"
        ) from error
