# Generate ohuman readble verview for results in excel format

import pandas as pd
from os import listdir
from os.path import isfile, join, abspath

class LLM_Result:
    def __init__(self):
        self.model = ""
        self.text = ""
        self.date = ""
        self.responsetime = ""
        self.response = ""
        self.type = ""

    def to_dict(self):
        return {
            "model": self.model,
            "text": self.text,
            "date": self.date,
            "responsetime": self.responsetime,
            "response": self.response,
            "type": self.type
        }

text_folder = "texts"
result_folder = "results"

resultlist = []

texts = [join(text_folder, f) for f in listdir(text_folder) if isfile(join(text_folder, f))]
results = [join(result_folder, f) for f in listdir(result_folder)]

data = {
    "Texts": texts
}

for result in results:
    files = [join(result, f) for f in listdir(result) if isfile(join(result, f))]

    for file in files:

        llm_result = LLM_Result()

        pure_filename = file.split("/")[-1]
        llm_result.date = (pure_filename.split("_")[2] + pure_filename.split("_")[3] + pure_filename.split("_")[4]).replace(".txt", "")
        llm_result.model = pure_filename.split("_")[0]
        llm_result.text = pure_filename.split("_")[1]
        llm_result.type = result.split("/")[1].split("_")[1]

        filehandle = open(file)
        filelines = filehandle.readlines()
        llm_result.response = str.join("\n", [l for l in filelines if l[0] != r"%" and l[0] != "\n"])

        llm_result.responsetime = filelines[-1].replace(r"%%Time consumed: ", "").replace(r" seconds %%", "")

        resultlist.append(llm_result)


#https://stackoverflow.com/questions/34997174/how-to-convert-list-of-model-objects-to-pandas-dataframe

df = pd.DataFrame.from_records([s.to_dict() for s in resultlist])

df.to_excel("results_Overview.xlsx")



