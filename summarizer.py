## built-in libs
import logging, os

## external libs
from transformers import BartForConditionalGeneration, BartTokenizer
import torch


## setup logging
lglvl: str = os.environ.get( 'LOGLEVEL', 'DEBUG' )
lglvldct = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO }
logging.basicConfig(
    level=lglvldct[lglvl],  # assigns the level-object to the level-key
    format='[%(asctime)s] %(levelname)s [%(module)s-%(funcName)s()::%(lineno)d] %(message)s',
    datefmt='%d/%b/%Y %H:%M:%S' )
log = logging.getLogger( __name__ )
log.debug( 'logging working' )


def load_model():
    """ Loads the model and tokenizer. 
        Called by manage_summarization(). """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer


def load_input_text():
    """ Loads the input text.
        Called by manage_summarization(). """
    with open('./test_files/obama.txt', 'r') as file:
    # with open('./test_files/org_description.txt', 'r') as file:
        input_text = file.read()
    return input_text


def chunk_text(tokenizer, input_text, max_length):
    """ Tokenizez the input text and splits into chunks. 
        Called by summarize_text(). """
    tokens = tokenizer.encode(input_text, add_special_tokens=False)
    chunk_size = max_length - 2  # Adjust for special tokens
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks


def summarize_text( model, tokenizer, input_text ):
    """ Summarizes the input text. 
        Called by manage_summarization(). """
    chunks = chunk_text(tokenizer, input_text, 1024)  # Adjust chunk size
    summaries = []

    for chunk in chunks:
        if len(chunk) > 1024:
            print( f'Chunk too long: ``{len(chunk)}`` tokens. Skipping.' )
            continue

        inputs = torch.tensor([chunk]).to(model.device)
        try:
            summary_ids = model.generate(
                inputs, 
                max_length=60,  
                min_length=30,  
                length_penalty=2.0,  
                no_repeat_ngram_size=2, 
                num_beams=4, 
                early_stopping=True
            )
            summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
        except RuntimeError as e:
            print( f'Error during summarization: ``{e}``' )

    full_summary = ' '.join(summaries)
    condensed_summary = full_summary.split('.')[0] + '.'
    return condensed_summary


def manage_summarization( input_text_filepath: str ):
    """ Manages the summarization process.
        Called by dundermain()."""
    model, tokenizer = load_model()
    input_text = load_input_text()
    summary = summarize_text(model, tokenizer, input_text)
    print( 'Summary:', summary )


if __name__ == "__main__":
    # model, tokenizer = load_model()
    # input_text = load_input_text()
    # summary = summarize_text(model, tokenizer, input_text)
    # print('Summary:', summary)

    log.debug( 'starting __main__' )
    log.info( '\n\nHHoag OCRed summarization ------------------------' )
    manage_summarization( './test_files/org_description_ocr.txt' )
    log.info( '\n\nObama speech summarization -----------------------' )
    manage_summarization( './test_files/obama_speech.txt' )
