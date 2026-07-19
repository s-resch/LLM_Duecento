from os import listdir
from os.path import isfile, join
from datetime import datetime
import time
from LLM_Duecento_Inference import GwdgInference, OllamaInference, TransformerInference

gwdgInference = GwdgInference()
ollamaInference = OllamaInference()
transformerInference = TransformerInference()

# Attention: On Ubuntu always use venv, otherwise we cannot install all packages
# So, we have to use venv and "source LLM_Duecento_Venv/bin/activate", after usage we can use "deactivate"

# ministral needs ollama 0.13.0

model_dict = {
    'dante': ollamaInference,
    'mistral': ollamaInference,
    'ministral-3:3b': ollamaInference,
    'cogito:8b': ollamaInference,
    'meta-llama-3.1-8b-instruct': gwdgInference,
    'openai-gpt-oss-120b': gwdgInference,
    'gemma-3-27b-it': gwdgInference,
    'llama-3.3-70b-instruct': gwdgInference,
    'mistral-large-instruct': gwdgInference,
    'teuken-7b-instruct-research': gwdgInference,
}

text_folder = "texts"
result_folder = "results"

texts = [join(text_folder, f) for f in listdir(text_folder) if isfile(join(text_folder, f))]

currentDateString = datetime.now().strftime("%d_%m_%y")

for model, inference in model_dict.items():
    for text in texts:
        text_analysis = open(text).read()
        text_name = text.split("/")[1].split(".")[0]
        model_name = model
        if model_name.count("/") > 0:
            model_name = model.split("/")[1]
        with open(join(result_folder, model_name + "_" + text_name + "_" + currentDateString + ".txt"), "w", encoding="utf-8") as f:
            f.write("%%Started: " +
                    datetime.now().strftime("%d/%m/%Y, %H:%M:%S") + "%%\n\n")
            time_start = time.perf_counter()

            result = inference.inference(model, text_analysis)
            f.write(result)

            time_end = time.perf_counter()
            time_duration = time_end - time_start
            f.write("%%Time consumed: " + str(time_duration) + " seconds %%")
