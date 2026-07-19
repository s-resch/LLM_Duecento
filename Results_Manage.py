# Generate overview for results

import pandas as pd
from os import listdir
from os.path import isfile, join, abspath
from abc import ABC, abstractmethod
from sklearn.metrics import jaccard_score
from rouge_score import rouge_scorer
import string
from bert_score import BERTScorer
import os
from huggingface_hub import snapshot_download
import pickle
from datetime import datetime


class Result(ABC):
    def __init__(self):
        self.text = ""
        self.response = ""
        self.type = ""


class Human_Result(Result):
    def __init__(self):
        super().__init__()


class LLM_Result(Result):
    def __init__(self):
        self.model = ""
        self.date = ""
        self.responsetime = ""
        self.jaccard = 0
        self.rouge = []
        self.bertscore = {}
        self.bertscore_custom = {}

    def to_dict(self):
        return {
            "model": self.model,
            "text": self.text,
            "date": self.date,
            "responsetime": self.responsetime,
            "response": self.response,
            "type": self.type,
            "jaccard": self.jaccard,
            "rouge1_p": self.rouge and self.rouge["rouge1"] and self.rouge["rouge1"].precision,
            "rouge1_r": self.rouge and self.rouge["rouge1"] and self.rouge["rouge1"].recall,
            "rouge1_f1": self.rouge and self.rouge["rouge1"] and self.rouge["rouge1"].fmeasure,
            "rouge2_p": self.rouge and self.rouge["rouge2"] and self.rouge["rouge2"].precision,
            "rouge2_r": self.rouge and self.rouge["rouge2"] and self.rouge["rouge2"].recall,
            "rouge2_f1": self.rouge and self.rouge["rouge2"] and self.rouge["rouge2"].fmeasure,
            "rougeL_p": self.rouge and self.rouge["rougeL"] and self.rouge["rougeL"].precision,
            "rougeL_r": self.rouge and self.rouge["rougeL"] and self.rouge["rougeL"].recall,
            "rougeL_f1": self.rouge and self.rouge["rougeL"] and self.rouge["rougeL"].fmeasure,
            "bertscore_p": self.bertscore and self.bertscore["precision"].item(),
            "bertscore_r": self.bertscore and self.bertscore["recall"].item(),
            "bertscore_f1": self.bertscore and self.bertscore["f1"].item(),
            "bertscore_custom_p": self.bertscore_custom and self.bertscore_custom["precision"].item(),
            "bertscore_custom_r": self.bertscore_custom and self.bertscore_custom["recall"].item(),
            "bertscore_custom_f1": self.bertscore_custom and self.bertscore_custom["f1"].item()
        }


class LLM_Result_Manager:
    def __init__(self, name=""):
        self.llm_results = []
        self.references = []
        self.path_delimiter = "/"
        self.name = name

    def save(self):
        path = "resultmanager"
        if not os.path.exists(path):
            os.makedirs(path)
        file = open(os.path.join(path, "rm_" + self.name + datetime.now().strftime("%d_%m_%y") + ".pkl"), "wb")
        pickle.dump(self, protocol=4, file=file)
        file.close()

    @classmethod
    def load(cls, file):
        f = open(file, mode="rb")
        return pickle.load(f)

    def gather(self, result_folder, ref_folder, type):
        self.gather_results(result_folder, type)
        self.gather_references(ref_folder, type)

    def gather_results(self, result_folder, result_type):
        results = [join(result_folder, f) for f in listdir(
            result_folder) if result_type in f]

        for result in results:
            files = [join(result, f)
                     for f in listdir(result) if isfile(join(result, f))]

            for file in files:

                llm_result = LLM_Result()

                pure_filename = file.split(self.path_delimiter)[-1]
                llm_result.date = (pure_filename.split("_")[
                    2] + pure_filename.split("_")[3] + pure_filename.split("_")[4]).replace(".txt", "")
                llm_result.model = pure_filename.split("_")[0]
                llm_result.text = pure_filename.split("_")[1]
                llm_result.type = result.split(self.path_delimiter)[1].split("_")[1]

                filehandle = open(file, encoding="utf-8")
                filelines = filehandle.readlines()
                llm_result.response = [
                    l.strip() for l in filelines if l != "" and l != "\n" and l != " "]

                self.llm_results.append(llm_result)

    def gather_references(self, ref_folder, type):
        references = [join(ref_folder, f)
                      for f in listdir(ref_folder) if type in f]

        for ref in references:
            files = [join(ref, f)
                     for f in listdir(ref) if isfile(join(ref, f))]

            for file in files:

                human_result = Human_Result()

                pure_filename = file.split(self.path_delimiter)[-1]
                human_result.text = pure_filename.replace(".txt", "")
                human_result.type = type

                filehandle = open(file, encoding="utf-8")
                filelines = filehandle.readlines()
                human_result.response = [
                    l.strip() for l in filelines if l != "" and l != "\n" and l != " "]

                self.references.append(human_result)

    def normalize_keywords(self, keywords):
        return_keywords = [kw.strip().lower()for kw in keywords]
        return_keywords.sort()
        return return_keywords

    def normalize_keywords_bertscore(self, keywords):
        return_keywords = [kw.strip().lower()for kw in keywords]
        return_keywords.sort()
        return_string = str.join("; ", return_keywords)
        return return_string

    def normalize_summary(self, summary):
        return_summary = str.join(" ", summary).strip().lower()
        return_summary = return_summary.translate(
            str.maketrans('', '', string.punctuation))
        return return_summary

    def binarize(self, labels, instance):
        return [1 if label in instance else 0 for label in labels]

    def calculate_jaccard(self):
        for llm_result in self.llm_results:
            for human_result in self.references:
                if llm_result.text == human_result.text and llm_result.type == human_result.type:
                    normalized_llm_keywords = self.normalize_keywords(
                        llm_result.response)
                    normalized_human_keywords = self.normalize_keywords(
                        human_result.response)
                    labels = sorted(
                        set(normalized_human_keywords + normalized_llm_keywords))

                    binarized_llm_keywords = self.binarize(
                        labels, normalized_llm_keywords)
                    binarized_human_keywords = self.binarize(
                        labels, normalized_human_keywords)

                    llm_result.jaccard = jaccard_score(
                        [binarized_human_keywords], [binarized_llm_keywords], average='samples')

    def calculate_rouge(self):
        for llm_result in self.llm_results:
            for human_result in self.references:
                if llm_result.text == human_result.text and llm_result.type == human_result.type:
                    normalized_llm_summary = self.normalize_summary(
                        llm_result.response)
                    normalized_human_summary = self.normalize_summary(
                        human_result.response)
                    scorer = rouge_scorer.RougeScorer(
                        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
                    llm_result.rouge = scorer.score(
                        normalized_human_summary, normalized_llm_summary)

    def calculate_bertscore_summary(self, model="bert-base-multilingual-cased", layers=9):
        baseline_path = None
        if (model != "bert-base-multilingual-cased"):
            modelname = model
            model = os.path.join("models", modelname)
            self.checkModelLocally(modelname, model)
            baselinefile = (modelname.split("/")[-1]) + ".tsv"
            baseline_path = os.path.join("config", baselinefile)
        scorer = BERTScorer(lang="it", model_type=model, num_layers=layers, rescale_with_baseline=True, baseline_path=baseline_path)
        for llm_result in self.llm_results:
            for human_result in self.references:
                if llm_result.text == human_result.text and llm_result.type == human_result.type:
                    normalized_llm_summary = self.normalize_summary(
                        llm_result.response)
                    normalized_human_summary = self.normalize_summary(
                        human_result.response)
                    P, R, F1 = scorer.score(
                        [normalized_human_summary], [normalized_llm_summary])
                    result_dict = {
                        "precision": P,
                        "recall": R,
                        "f1": F1
                    }
                    if model != "bert-base-multilingual-cased":
                        llm_result.bertscore_custom = result_dict
                    else:
                        llm_result.bertscore = result_dict

    def calculate_bertscore_keywords(self, model="bert-base-multilingual-cased", layers=9):
        baseline_path = None
        if (model != "bert-base-multilingual-cased"):
            modelname = model
            model = os.path.join("models", modelname)
            self.checkModelLocally(modelname, model)
            baselinefile = (modelname.split("/")[-1]) + ".tsv"
            baseline_path = os.path.join("config", baselinefile)
        scorer = BERTScorer(lang="it", model_type=model, num_layers=layers, rescale_with_baseline=True, baseline_path=baseline_path)
        for llm_result in self.llm_results:
            for human_result in self.references:
                if llm_result.text == human_result.text and llm_result.type == human_result.type:
                    normalized_llm_keywords = self.normalize_keywords_bertscore(
                        llm_result.response)
                    normalized_human_keywords = self.normalize_keywords_bertscore(
                        human_result.response)
                    P, R, F1 = scorer.score(
                        [normalized_human_keywords], [normalized_llm_keywords])
                    result_dict = {
                        "precision": P,
                        "recall": R,
                        "f1": F1
                    }
                    if model != "bert-base-multilingual-cased":
                        llm_result.bertscore_custom = result_dict
                    else:
                        llm_result.bertscore = result_dict

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
