from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def summarize_text(model, tokenizer, input_text):
    inputs = tokenizer.encode("summarize: " + input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    model, tokenizer = load_model()
    input_text = input('Enter text to summarize: ')
    summary = summarize_text(model, tokenizer, input_text)
    print('Summary:', summary)


# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch

# def load_model():
#     tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
#     model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B')
#     return model, tokenizer

# def summarize_text(model, tokenizer, input_text):
#     inputs = tokenizer.encode('Summarize: ' + input_text, return_tensors='pt')
#     summary = model.generate(inputs, max_length=50, num_return_sequences=1)
#     return tokenizer.decode(summary[0], skip_special_tokens=True)

# if __name__ == "__main__":
#     model, tokenizer = load_model()
#     input_text = input('Enter text to summarize: ')
#     summary = summarize_text(model, tokenizer, input_text)
#     print('Summary:', summary)
