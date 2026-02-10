from langchain_huggingface import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="distilgpt2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 80,
        "temperature": 0.7
    }
)

print(llm.invoke("What is the capital of India?"))
