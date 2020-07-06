import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from run_generation import sample_sequence
from yt_encoder import YTEncoder
from transformers import GPT2LMHeadModel
import threading
import regex as re

from os import environ
device = environ.get('DEVICE', 'cuda:0')

flavor_id = device + environ.get('INSTANCE', ':0')
from tendo import singleton
me = singleton.SingleInstance(flavor_id=flavor_id)

import logging

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(filename=f"logs/{hash(flavor_id)}.log", level=logging.INFO)
logger = logging.getLogger(__name__)


BASE_PATH = '../../content/drive/My Drive/Colab Notebooks/ru_trfs_inference/ckpts/'
model_paths = ['medium',
               'gradual_unfreeze',
               'all_unfreeze',
               'noy']
tokenizer = YTEncoder.from_pretrained(BASE_PATH+model_paths[0])
models = [GPT2LMHeadModel.from_pretrained(BASE_PATH+m_path).to(device).eval() for m_path in model_paths]

from apex import amp
models = amp.initialize(models, opt_level='O2')
print(f'{len(models)} models loaded for inference. URLs: {model_paths}')
def get_sample(
                model,
                prompt,
                length:int,
                num_samples:int,
                allow_linebreak:bool,
                temperature:float,
                top_p:float,
                top_k:int):
    logger.info(prompt)

    filter_n = tokenizer.encode('\n')[-1:]
    filter_single = [1] + tokenizer.encode('[')[-1:] + tokenizer.encode('(')[-1:]
    filter_single += [] if allow_linebreak else filter_n

    context_tokens = tokenizer.encode(prompt)
    out = sample_sequence(
        model=model,
        context=context_tokens,
        length=length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        device=device,
        filter_single=filter_single,
        filter_double=filter_n,
        num_samples=num_samples,
    ).to('cpu')

    prompt = tokenizer.decode(context_tokens)
    len_prompt = len(prompt)

    replies = [out[item, :].tolist() for item in range(len(out))]
    text = [tokenizer.decode(item)[len_prompt:] for item in replies]
    reg_text = [re.match(r'[\w\W]*[\.!?]\n', item) for item in text]
    reg_text2 = [re.match(r'[\w\W]*[\.!?]', item) for item in text]
    result = [reg_item[0] if reg_item else reg_item2[0] if reg_item2 else item for reg_item, reg_item2, item in zip(reg_text, reg_text2, text)]
    logger.info(result)
    return result

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Schema

app = FastAPI(title="Russian GPT-2", version="0.1",)
app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

lock = threading.RLock()

class Prompt(BaseModel):
    prompt:str = Schema(..., max_length=3000, title='Model prompt')
    length:int = Schema(150, ge=1, le=200, title='Number of tokens generated in each sample')
    num_samples:int = Schema(3, ge=1, le=3, title='Number of samples generated')
    allow_linebreak:bool = Schema(False, title='Allow linebreak in a sample')
    temperature:float = Schema(1.0, title='temperature')
    top_p:float = Schema(0.9, title='top_p')
    top_k:int = Schema(0, title='top_k')



@app.post("/{model_path}/")
def gen_sample(prompt: Prompt, model_path: str ):
    with lock:
        rmodels = [model for path, model in zip(model_paths,models) if path==model_path]
        return {"replies": get_sample(
                                    rmodels[0],
                                    prompt.prompt,
                                    prompt.length,
                                    prompt.num_samples,
                                    prompt.allow_linebreak,
                                    prompt.temperature,
                                    prompt.top_p,
                                    prompt.top_k)}

class PromptPoetry(BaseModel):
    prompt:str = Schema(..., max_length=3000, title='Model prompt')
    length:int = Schema(15, ge=1, le=150, title='Number of tokens generated in each sample')

@app.post("/gpt2_poetry/")
def gen_sample(prompt: PromptPoetry):
    with lock:
        return {"replies": get_sample(poetry_model, prompt.prompt, prompt.length, 1, True)}

@app.get("/health")
def healthcheck():
    return True
