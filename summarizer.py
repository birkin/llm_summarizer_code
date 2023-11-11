from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def load_input_text():
    with open('./test_files/obama.txt', 'r') as file:
        input_text = file.read()
    return input_text

def load_model():
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer

def chunk_text(tokenizer, input_text, max_length):
    # Tokenize the input text and split into chunks
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    chunk_size = max_length - 2  # for special tokens [CLS] and [SEP]
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks

def summarize_text(model, tokenizer, input_text):
    chunks = chunk_text(tokenizer, input_text, 1024)
    summaries = []

    for chunk in chunks:
        inputs = torch.tensor([chunk])
        # Estimate appropriate generation lengths
        estimated_max_length = int(100 * len(chunk) / len(tokenizer.encode(input_text)))
        estimated_min_length = max(int(estimated_max_length * 0.75), 40)  # Ensure a minimum length

        summary_ids = model.generate(inputs, max_length=estimated_max_length, min_length=estimated_min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
        summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
    full_summary = ' '.join(summaries)
    return full_summary

# def summarize_text(model, tokenizer, input_text):
#     # Split text into chunks
#     chunks = chunk_text(tokenizer, input_text, 1024)
#     summaries = []

#     for chunk in chunks:
#         # Convert chunk to torch tensor
#         inputs = torch.tensor([chunk])
#         # Generate summary for the chunk
#         summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
#         # Decode and add the summary to the list
#         summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
    
#     # Concatenate all chunk summaries
#     full_summary = ' '.join(summaries)
#     return full_summary

if __name__ == "__main__":
    model, tokenizer = load_model()
    input_text = load_input_text()
    summary = summarize_text(model, tokenizer, input_text)
    print('Summary:', summary)
