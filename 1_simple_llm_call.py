from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# Simple one-line prompt
prompt = PromptTemplate.from_template("{question}")

model_gen = HuggingFaceEndpoint(
    # repo_id="Qwen/Qwen2.5-7B-Instruct",
    repo_id="google/gemma-2-2b-it",
    # repo_id="openai/gpt-oss-20b",
    # repo_id="MiniMaxAI/MiniMax-M2",
    # repo_id="meta-llama/Llama-3.1-70B-Instruct",
    # repo_id="moonshotai/Kimi-K2-Thinking",
    task="text-generation",
    max_new_tokens=200,
    do_sample=False,
    temperature=0.2,
    device_map=None,       # Force remote inference
)
generator_llm = ChatHuggingFace(llm=model_gen)
parser = StrOutputParser()

# Chain: prompt → model → parser
chain = prompt | generator_llm | parser

# Run it
result = chain.invoke({"question": "What is the capital of Pakistan?"})
print(result)
