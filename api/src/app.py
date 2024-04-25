from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from summarizer import Summarizer

#PARAMETERS
model="sshleifer/distilbart-cnn-12-6"
upper_relative_treshold = 0.65
lower_relative_treshhold = 0.2
num_beams = 3
do_sample=True

# MODEL INSTANTIATION
summ = Summarizer(
        model=model, 
        upper_relative_treshold=upper_relative_treshold,
        lower_relative_treshhold=lower_relative_treshhold,
        num_beams=num_beams,
        do_sample=do_sample)



transcription = """
The initials gpt stand for generative pre-trained transformer. So that first word is straightforward enough, these are bots that generate new text. 
Pre-trained refers to how the model went through a process of learning from a massive amount of data and the prefix insinuates that there's more room to fine 
tune it on specific tasks with additional training. But the last word, that's the real key piece. A transformer is a specific kind of neural network, a machine 
learning model, and it's the core invention underlying the current boom in AI. What I want to do with this video and the following chapters is go through a visually 
driven explanation for what actually happens inside a transformer. We're going to follow the data that flows through it and go step by step. There are many different kinds 
of models that you can build using transformers. Some models take in audio and produce a transcript. This sentence comes from a model going the other way around, producing 
synthetic speech just from text. All those tools that took the world by storm in 2022, like dolly and mid-journey that take in a text description and produce an image are 
based on transformers. And even if I can't quite get it to understand what a pie creature is supposed to be, I'm still blown away that this kind of thing is even remotely possible
"""

class InputUrl(BaseModel):
    youtube_url: str

app = FastAPI()

@app.post("/summarization/")
async def echo_string(input_string: InputUrl):
    return {
        "summary": summ.summarize(transcription)  
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
