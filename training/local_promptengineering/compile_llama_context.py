from google import genai
from dotenv import load_dotenv
import os

module_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv()
client = genai.Client()

if __name__ == "__main__":
    user_input = ""
    print("Welcome to DebateLord. What would you like to debate about today?")
    user_input = str(input())

    training_data = ""
    
    with open(os.path.join(module_dir, "./gemini_engineered_context.md"), "r+") as context_file:
        context = context_file.read()
        pre_data_ctx = context[0:context.index("[Insert here]")]
        post_data_ctx = context[context.index("[Insert here]"):len(context)]

        # Change this to get all files in the directory instead of a hardcoded 10
        for findex in range(10):
            new_content = open(os.path.join(module_dir, f"../local_finetuning/training_data/debate_{findex}_patterns.json")).read()
            training_data += f"\n{new_content}"
        
        context_file.close()

    while user_input != "STOP":
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                {
                    "parts": [
                        {"text" : f"SYSTEM:\n{pre_data_ctx}\n{training_data}\n{post_data_ctx}\n\nUSER: {user_input}"}
                    ]
                }
            ]
        )

        print(f"\n{response.text}\n")
        user_input = str(input())
    
    print("Done so soon? Can't blame you. Debate Lord, OUT!\n")

