import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

CACHE_DIR = "/content/drive/MyDrive/LLM_RAG_Bot/models"

class ChatModel:
    def __init__(self, model_id: str = "google/gemma-2b-it", device="cuda"):


        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",
                                                          quantization_config=quantization_config,
                                                          cache_dir=CACHE_DIR)


        self.model.eval()
        self.chat = []
        self.device = device

    def inference(self, question: str, context: str = None, max_new_tokens: int = 250):

        if context == None or context == "":
            prompt = f"""Give a detailed answer to the question. Question: {question}"""
        else:
            prompt = f"""Using the information contained from the context, give a detailed answer to the question. Do
            not add any extra information . Context: {context}.
Question: {question}"""

        chat = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )
    
        inputs = self.tokenizer.encode(
            formatted_prompt, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        response = response[len(formatted_prompt) :]  
        response = response.replace("<eos>", "") 

        return response