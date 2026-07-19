from abc import ABC, abstractmethod
import os
from dotenv import load_dotenv
from openai import OpenAI
from ollama import Client as ollamaClient
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download


class Inference(ABC):
    def __init__(self):
        self.url = ""
        
        self.system_prompt = """Sei specializzato in letteratura italiana. 
        Ti occupi di testi in lingua italiana, li comprendi e li utilizzi nel tuo lavoro."""

        # self.task = """Prendi il testo sottostante e riassumi il suo contenuto. Non scrivere più di 100 parole."""
        self.task = """Prendi il testo sottostante e trova parole chiave che descrivano in modo panoramico il contenuto del testo fornito. Trova tra 4 e 6 parole chiave."""
        
        self.task_prompt = """Il tuo compito: """ + self.task + """

        Esegui il compito al fine di fornire informazioni a persone esperte in letteratura italiana che non hanno letto il testo fornito. 
        
        Non devi utilizzare alcuna informazione esterna! Utilizza solo le tue conoscenze interne e la tua comprensione della lingua. Ecco il testo: \n {text}"""
        
        self.client = None
        self.temperature = 0.7

    @abstractmethod
    def inference(self, model="", text=""):
        pass


class GwdgInference(Inference):
    def __init__(self):
        super().__init__()
        load_dotenv()
        self.api_key = os.getenv("SAIA_API_KEY")
        self.url = "https://chat-ai.academiccloud.de/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.url)

    def inference(self, model, text):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system", "content": self.system_prompt
                },
                {
                    "role": "user", "content": self.task_prompt.format(text=text)
                }],
            model=model,
            temperature=self.temperature
        )
        response = ""
        for choice in chat_completion.choices:
            response += choice.message.content.strip()
            response += "\n"

        return response


class OllamaInference(Inference):
    def __init__(self):
        super().__init__()
        self.url = "http://localhost:11434"
        self.client = ollamaClient(host=self.url)

    def inference(self, model, text):
        chat_completion = self.client.chat(
            messages=[
                {
                    "role": "system", "content": self.system_prompt
                },
                {
                    "role": "user", "content": self.task_prompt.format(text=text)
                }],
            model=model,
            options={"temperature": self.temperature}
        )
        response = chat_completion['message']['content']

        return response


class TransformerInference(Inference):
    def __init__(self):
        super().__init__()
        self.modelBaseDir = ""

    def inference(self, model, text):
        modeldir = os.path.join(self.modelBaseDir, model)
        self.checkModelLocally(model, modeldir)

        device = "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            modeldir, device_map="auto", load_in_8bit=True)
        tokenizer = AutoTokenizer.from_pretrained(modeldir)

        messages = [
            {
                "role": "system", "content": self.system_prompt
            },
            {
                "role": "user", "content": self.task_prompt.format(text=text)
            }
        ]

        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)
        model.to(device)

        generated_ids = model.generate(
            input_ids=model_inputs, max_new_tokens=300, do_sample=True, temperature=self.temperature)
        decoded = tokenizer.batch_decode(generated_ids)

        return decoded[0]

    def checkModelLocally(self, model, modeldir):
        message = ""
        # Check if the model is already locally available
        if not os.path.exists(modeldir):
            # Download the model and save it locally
            snapshot_download(repo_id=model, local_dir=modeldir)
            message = "Model downloaded and saved locally."
        else:
            message = "Model is already locally available."

        return message
