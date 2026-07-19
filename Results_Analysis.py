from Results_Manage import LLM_Result, LLM_Result_Manager, Human_Result

resultManagerPC = LLM_Result_Manager("pc")
resultManagerPC.gather(r"results", r"references", "ParoleChiave")
resultManagerPC.calculate_jaccard()
resultManagerPC.calculate_bertscore_keywords()
resultManagerPC.save()

resultManagerPC_DBMDZ = LLM_Result_Manager("pc_dbmdz")
resultManagerPC_DBMDZ.gather(r"results", r"references", "ParoleChiave")
# Use 10 layers like in https://github.com/gsarti/it5/blob/main/compute_scores.ipynb
resultManagerPC_DBMDZ.calculate_bertscore_keywords(
    model="dbmdz/bert-base-italian-xxl-uncased", layers=10)
resultManagerPC_DBMDZ.save()

resultManagerSummary = LLM_Result_Manager("rias")
resultManagerSummary.gather(r"results", r"references", "Riassunto")
resultManagerSummary.calculate_rouge()
resultManagerSummary.calculate_bertscore_summary()
resultManagerSummary.save()

resultManagerSummary_DBMDZ = LLM_Result_Manager("rias_dbmdz")
resultManagerSummary_DBMDZ.gather(r"results", r"references", "Riassunto")
# Use 10 layers like in https://github.com/gsarti/it5/blob/main/compute_scores.ipynb
resultManagerSummary_DBMDZ.calculate_bertscore_summary(
    model="dbmdz/bert-base-italian-xxl-uncased", layers=10)
resultManagerSummary_DBMDZ.save()




