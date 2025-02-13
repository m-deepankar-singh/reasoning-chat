import sys
import os
sys.path.append(os.path.abspath("."))

import asyncio
import logging
from app.services.reasoning_engine import reasoning_engine

# Configure logging with a file handler to write logs to logs/app.log
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/app.log")
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

async def run_debug_test():
    prompt = "This is a debug test prompt for reasoning."
    # Enable reasoning with no uploaded files (empty images list)
    try:
        result = await reasoning_engine.get_reasoning(prompt, enable_reasoning=True, images=[])
        print("Test Reasoning Result:")
        print(result)
    except Exception as e:
        print("Error during reasoning test:", e)

if __name__ == "__main__":
    asyncio.run(run_debug_test())