import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from statistics import mean, stdev
from math import sqrt

def cohenD(group1, group2):
    return (mean(group1) - mean(group2)) / (sqrt((stdev(group1) ** 2 + stdev(group2) ** 2) / 2))

# Read data
dateipfad = "resultsexport/Metriken_Rias_Cleaned.csv"
df = pd.read_csv(
    dateipfad,
    delimiter=";",
    decimal=",",
    encoding="utf-8"
)

# Define mode (model or text)
mode = 'model'

# Only keep needed values
df_filtered_custom_bert = df[[mode, 'AVG_BERT_custom_f1']] 
df_filtered_bert = df[[mode, 'AVG_BERT_f1']] 
df_filtered_rouge1 = df[[mode, 'AVG_rouge1_f1']] 
df_filtered_rouge2 = df[[mode, 'AVG_rouge2_f1']] 
df_filtered_rougeL = df[[mode, 'AVG_rougeL_f1']] 

# Group values
groups_custom_bert = df_filtered_custom_bert.groupby(mode)['AVG_BERT_custom_f1'].apply(list).to_dict()
groups_bert = df_filtered_bert.groupby(mode)['AVG_BERT_f1'].apply(list).to_dict()
groups_rouge1 = df_filtered_rouge1.groupby(mode)['AVG_rouge1_f1'].apply(list).to_dict()
groups_rouge2 = df_filtered_rouge2.groupby(mode)['AVG_rouge2_f1'].apply(list).to_dict()
groups_rougeL = df_filtered_rougeL.groupby(mode)['AVG_rougeL_f1'].apply(list).to_dict()

# Get labels
labels_custom_bert = list(groups_custom_bert.keys())
labels_bert = list(groups_bert.keys())
labels_rouge1 = list(groups_rouge1.keys())
labels_rouge2 = list(groups_rouge2.keys())
labels_rougeL = list(groups_rougeL.keys())

# Save results into file
file = open(f"cohen_rias_{mode}.txt", "w", encoding="utf-8")

for i in range(len(labels_custom_bert)):
    for j in range(i + 1, len(labels_custom_bert)):
        label1 = labels_custom_bert[i]
        label2 = labels_custom_bert[j]
        # Cohens D
        cohen = cohenD(groups_custom_bert[label1], groups_custom_bert[label2])
        file.write(f"Compare Custom BERT {label1} vs {label2}: Cohen's D={cohen:.3f}\n")
            

for i in range(len(labels_bert)):
    for j in range(i + 1, len(labels_bert)):
        label1 = labels_bert[i]
        label2 = labels_bert[j]
        # Cohens D
        cohen = cohenD(groups_bert[label1], groups_bert[label2])
        file.write(f"Compare BERT {label1} vs {label2}: Cohen's D={cohen:.3f}\n")

for i in range(len(labels_rouge1)):
    for j in range(i + 1, len(labels_rouge1)):
        label1 = labels_rouge1[i]
        label2 = labels_rouge1[j]
        # Cohens D
        cohen = cohenD(groups_rouge1[label1], groups_rouge1[label2])
        file.write(f"Compare Rouge1 {label1} vs {label2}: Cohen's D={cohen:.3f}\n")

for i in range(len(labels_rouge2)):
    for j in range(i + 1, len(labels_rouge2)):
        label1 = labels_rouge2[i]
        label2 = labels_rouge2[j]
        # Cohens D
        cohen = cohenD(groups_rouge2[label1], groups_rouge2[label2])
        file.write(f"Compare Rouge2{label1} vs {label2}: Cohen's D={cohen:.3f}\n")

for i in range(len(labels_rougeL)):
    for j in range(i + 1, len(labels_rougeL)):
        label1 = labels_rougeL[i]
        label2 = labels_rougeL[j]
        # Cohens D
        cohen = cohenD(groups_rougeL[label1], groups_rougeL[label2])
        file.write(f"Compare RougeL {label1} vs {label2}: Cohen's D={cohen:.3f}\n")

