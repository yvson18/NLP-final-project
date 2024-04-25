from transformers import pipeline

class Summarizer:
    def __init__(self,
        model: str,         
        upper_relative_treshold: float, 
        lower_relative_treshhold: float,
        num_beams: int,
        do_sample: bool
    ):
        self.pipe = pipeline("summarization", model=model, tokenizer=model)
        self.upper_relative_treshold = upper_relative_treshold
        self.lower_relative_treshhold = lower_relative_treshhold
        self.num_beams = num_beams
        self. do_sample = do_sample
        
    def summarize(self, text: str):
        num_words = len(text.split())
        text = text.strip()
        result = self.pipe(text, 
                    max_length=int(self.upper_relative_treshold * num_words),
                    min_new_tokens=int(self.lower_relative_treshhold * num_words), 
                    do_sample=self.do_sample, 
                    num_beams=self.num_beams)
        
        
        return result[0]['summary_text']