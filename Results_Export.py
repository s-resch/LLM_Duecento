from Results_Manage import LLM_Result, LLM_Result_Manager, Human_Result
import pandas as pd
from os import listdir, makedirs
from os.path import isfile, join, abspath, exists
from collections import defaultdict
from rouge_score import rouge_scorer

rm_folder = "resultmanager"

rms = [join(rm_folder, f) for f in listdir(rm_folder)]

for rm in rms:
    resultmanager = LLM_Result_Manager.load(rm)

    #df = pd.DataFrame.from_records([s.to_dict() for s in resultmanager.llm_results])

    results_as_dict = [s.to_dict() for s in resultmanager.llm_results]

    # if "pc" in resultmanager.name:
    #     if "dbmdz" in resultmanager.name:
    # # Attributes average should be calculated for
    #         attrs_for_avg = ["jaccard", "bertscore_custom_p", "bertscore_custom_r", "bertscore_custom_f1"]
    #     else:
    #         attrs_for_avg = ["jaccard", "bertscore_p", "bertscore_r", "bertscore_f1"]
    # else:
    #     if "dbmdz" in resultmanager.name:
    #         attrs_for_avg = ["rouge1_p", "rouge1_r", "rouge1_f1", "rouge2_p", "rouge2_r", "rouge2_f1", "rougeL_p", "rougeL_r", "rougeL_f1", "bertscore_custom_p", "bertscore_custom_r", "bertscore_custom_f1"]
    #     else:
    #         attrs_for_avg = ["rouge1_p", "rouge1_r", "rouge1_f1", "rouge2_p", "rouge2_r", "rouge2_f1", "rougeL_p", "rougeL_r", "rougeL_f1", "bertscore_p", "bertscore_r", "bertscore_f1"]

    # # Group values
    # grouped = defaultdict(list)
    # for result in results_as_dict:
    #     key = (result["model"], result["text"])
    #     grouped[key].append(result)

    # # Calculate averages
    # resultlist = []
    # for (a, b), items in grouped.items():
    #     row = {"Model": a, "Text": b}
    #     for key in attrs_for_avg:  # attrs_for_avg contains keys average should be calculated for
    #         first_item = items[0]
    #         if key in first_item:
    #             values = [item[key] for item in items]
    #             avg = sum(values) / len(values)
    #             row[f"{key}_avg"] = avg
    #         resultlist.append(row)


    filename_raw = rm.split("/")[1].replace("pkl", "xlsx")
    #filename = rm.split("/")[1].replace(".pkl", "_grouped_avg.xlsx")

    pathname = "resultsexport"
    if not exists(pathname):
        makedirs(pathname)

    #filename = join(pathname, filename)
    filename_raw = join(pathname, filename_raw)

    #df = pd.DataFrame.from_records(resultlist)
    df_raw = pd.DataFrame.from_records(results_as_dict)

    #df.to_excel(filename)
    df_raw.to_excel(filename_raw)

