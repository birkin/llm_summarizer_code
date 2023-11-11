# Summarization experimentation

on this page...
- Purpose
- Outcome
    - Quality
    - Time
- Usage
- Output

---


## Purpose

With upcoming Hall-Hoag work, wanted to explore summarization using a pretrained general large-language-model (LLM) for possible inclusion into processing-pipeline.

---


## Outcome

### Quality

Not satisfied. Most of the online research indicates that summarizers work by yielding representative excerpts. I got this code to do that; it now yields roughly a single representative sentence.

However, I didn't want 'excerpts' as summarization. The current code, for the Obama first inaugural speech, yields:
```
SUMMARIZATION-EXCERPT, ``I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time.``
```

What I wanted was something more like: "Summary: The person describes thanking voters, and thanking opponents, and expresses hope for the future. It touches on themes of..." That's _why_ I tried this with an LLM. I'm pretty new at this, there are very likely combinations of models and settings I could use/tweak to do this.

Eventually! 🙂

### Time

This could _easily_ be something I'm not optimizing, but the summarization is _slow_. Running it on my Mac M1-Pro on either of the two test-files processes about 30-40 words-per-second.

---


## Usage
- assumes a virtual-environment is set up (at sibling level to code) and populated from `requirements.txt`
- $ cd /path/to/llm_summarizer_code/
- $ source ../env/bin/activate

Then...
- $ python ./summarizer.py  # will auto-run summarizer on the two test-files

...or...

- $ python ./summarizer.py  --input_path "/path/to/target_file.txt"

(TODO: enable url as argument)

---


## Output

```
HHoag OCRed summarization ------------------------
[11/Nov/2023 16:19:00] DEBUG [summarizer-load_input_text()::69] word_count, ``2462``
[11/Nov/2023 16:20:15] INFO [summarizer-manage_summarization()::29] SUMMARIZATION-EXCERPT, ``“1313” swarms with an executive elite, including NML, who sit on each others boards, commissions, committees and secretariats.``
[11/Nov/2023 16:20:15] DEBUG [summarizer-<module>()::152] ending dundermain, time elapsed: 79.33 seconds

Obama speech ----------------------------------------
[11/Nov/2023 16:13:47] DEBUG [summarizer-load_input_text()::69] word_count, ``2156``
[11/Nov/2023 16:14:38] INFO [summarizer-manage_summarization()::29] SUMMARIZATION-EXCERPT, ``I want to thank every American who participated in this election, whether you voted for the very first time or waited in line for a very long time.``
[11/Nov/2023 16:14:38] DEBUG [summarizer-<module>()::152] ending dundermain, time elapsed: 134.64 seconds
```

---