# src/llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load Flan-T5-Small (CPU-friendly)
MODEL_NAME = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

def generate_answer(query: str, context: str, max_length: int = 200) -> str:
    """
    Generate answer given a query and context using Flan-T5-Small.
    """
    input_text = f"Question: {query} Context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False
        )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
