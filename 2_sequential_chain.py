from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

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

chain = prompt1 | generator_llm | parser | prompt2 | generator_llm | parser

result = chain.invoke({'topic': 'Unemployment in India'})

print(result)
