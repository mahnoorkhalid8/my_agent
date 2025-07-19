import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from agents.run import RunConfig
import asyncio

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please set it in .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config =  RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

async def main():
    agent= Agent(
        name="Helpful Assisstant",
        instructions="Tou are a helpful assistant that can answer questions and help with tasks.",
        model=model
    )

    result = await Runner.run(starting_agent=agent, input="Hello, how are you? What is the capital of Pakistan?", run_config=config)
    print("\nCalling Agent\n")
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())