"""
File: engineered_debatelord.py
Description: Handles the engineering of DebateLord's prompt and context for Gemini API.

Author: Berkan Mertan
Email: berkm@ihmail.com
Copyright (c) 2025 Berkan Mertan. All rights reserved.

This software is provided "as is" without warranty of any kind.
Use at your own risk.
"""

import asyncio
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import compile_llama_context as ContextCompiler

load_dotenv()

client = genai.Client()

# Configuration for Live API to ground DebateLord in some google search reality
model = "gemini-live-2.5-flash-preview"
tools = [{'google_search': {}}]
config = {"response_modalities": ["TEXT"], "tools": tools}

# For smaller, less Python-generation based Gemini requests
# grounding_tool = types.Tool(
#     google_search=types.GoogleSearch()
# )
# config = types.GenerateContentConfig(
#     tools=[grounding_tool]
# )

# User input must keep being added onto to preserve context
raw_past_inputs = ""

def prepare_full_input(added_context: str, past_line: str) -> str:
    global raw_past_inputs
    raw_past_inputs += f"{past_line}\n"
    return f"\nPreviously said by user: {raw_past_inputs}\nUser just said: {added_context}"

async def main():
    past_input = ""
    user_input = ContextCompiler.initiate_conversation()
    
    while user_input != "STOP":
        past_input = user_input
        prompt = ContextCompiler.prepare_debate_context(prepare_full_input(user_input, past_input))
        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(turns={"parts": [{"text": prompt}]})

            async for chunk in session.receive():
                if chunk.server_content:
                    if chunk.text is not None:
                        print(chunk.text)

                    # The model might generate and execute Python code to use Search
                    model_turn = chunk.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.executable_code is not None:
                                print(part.executable_code.code)

                            if part.code_execution_result is not None:
                                print(part.code_execution_result.output)
        user_input = str(input())

if __name__ == "__main__":
    asyncio.run(main())