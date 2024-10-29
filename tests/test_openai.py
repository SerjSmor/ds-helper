from unittest import TestCase
from dotenv import load_dotenv

from openai import OpenAI


class TestOpenAi(TestCase):

    def test_setup(self):
        load_dotenv()
        client = OpenAI()

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": "Write a haiku about recursion in programming."
                }
            ]
        )
