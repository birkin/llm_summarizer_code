## built-in libs
import argparse, logging, os

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


## main controller function
def manage_summarization( input_text_filepath: str ):
    """ Manages the summarization process.
        Called by dundermain()."""
    model, tokenizer = load_model()
    input_text: str = load_input_text( input_text_filepath )
    summary: str = summarize_text(model, tokenizer, input_text)
    log.info( f'SUMMARIZATION-EXCERPT, ``{summary}``' )


def load_model():
    """ Loads the model and tokenizer. 
        Called by manage_summarization(). """
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
    return model, tokenizer


def load_input_text( input_text_filepath: str ):
    """ Loads the input text.
        Called by manage_summarization(). """
    with open( input_text_filepath, 'r' ) as file:
        input_text = file.read()
    return input_text


def summarize_text(model, tokenizer, input_text):
    """ Summarizes the input text. 
        Called by manage_summarization(). """
    pre_chunk_size = 1024  # Adjust chunk size for pre-chunking if necessary
    all_summaries = []

    if len(input_text) > pre_chunk_size:
        # Pre-chunk the text into smaller parts if it's too long
        pre_chunks = [input_text[i:i + pre_chunk_size] for i in range(0, len(input_text), pre_chunk_size)]
        
        for pre_chunk in pre_chunks:
            chunks = chunk_text(tokenizer, pre_chunk, 1024)
            summaries = generate_summary(model, tokenizer, chunks)
            all_summaries.extend(summaries)
    else:
        chunks = chunk_text(tokenizer, input_text, 1024)
        all_summaries = generate_summary(model, tokenizer, chunks)

    full_summary = ' '.join(all_summaries)
    condensed_summary = full_summary.split('.')[0] + '.'
    return condensed_summary


def chunk_text(tokenizer, input_text, max_length):
    """ Tokenizez the input text and splits into chunks. 
        Called by summarize_text(). """
    tokens = tokenizer.encode(input_text, add_special_tokens=False, truncation=True, max_length=max_length)
    chunk_size = max_length - 2  # Adjust for special tokens
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    return chunks


def generate_summary(model, tokenizer, chunks):
    """ Generates summaries for given chunks of text. 
        Called by summarize_text(). """
    summaries = []
    for chunk in chunks:
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
            print(f'Error during summarization: ``{e}``')
    return summaries


if __name__ == "__main__":
    log.debug( 'starting dundermain' )

    ## set up argparser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--input_path', type=str, help='Path to the input file')
    args = parser.parse_args()
    log.debug( f'args: {args}' )
    ## get input path
    input_path = args.input_path if args.input_path else ''
    log.debug( f'input_path: {input_path}' )

    ## if there is an input_path, use it, otherwise run the summarizer on the two test-files
    if input_path:
        log.info( f'\n\nSummarization for input filepath, ``{input_path}``...' )
        manage_summarization( input_path )
    else:
        log.info( '\n\nHHoag OCRed summarization ------------------------' )
        manage_summarization( './test_files/org_description_ocr.txt' )
        log.info( '\n\nObama speech summarization -----------------------' )
        manage_summarization( './test_files/obama_speech.txt' )

    log.debug( 'ending dundermain' )

    # log.info( '\n\nHHoag OCRed summarization ------------------------' )
    # manage_summarization( './test_files/org_description_ocr.txt' )
    # log.info( '\n\nObama speech summarization -----------------------' )
    # manage_summarization( './test_files/obama_speech.txt' )
