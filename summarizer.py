from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
    return model, tokenizer

def summarize_text(model, tokenizer, input_text):
    inputs = tokenizer.encode('Summarize: ' + input_text, return_tensors='pt')
    summary = model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(summary[0], skip_special_tokens=True)


if __name__ == "__main__":
    model, tokenizer = load_model()
    input_text = input('Enter text to summarize: ')
    summary = summarize_text(model, tokenizer, input_text)
    print('Summary:', summary)
