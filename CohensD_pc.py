import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu
from statistics import mean, stdev
from math import sqrt

def cohenD(group1, group2):
    return (mean(group1) - mean(group2)) / (sqrt((stdev(group1) ** 2 + stdev(group2) ** 2) / 2))

# Read data
dateipfad = "resultsexport/Metriken_PC.csv"
df = pd.read_csv(
    dateipfad,
    delimiter=";",
    decimal=",",
    encoding="utf-8"
)

# Define mode (model or text)
mode = 'text'

# Only keep needed values
df_filtered_custom_bert = df[[mode, 'AVG_BERT_custom_f1']]
df_filtered_bert = df[[mode, 'AVG_BERT_f1']] 
df_filtered_jaccard = df[[mode, 'AVG_jaccard']]

# Group values
groups_custom_bert = df_filtered_custom_bert.groupby(mode)['AVG_BERT_custom_f1'].apply(list).to_dict()
groups_bert = df_filtered_bert.groupby(mode)['AVG_BERT_f1'].apply(list).to_dict()
groups_jaccard = df_filtered_jaccard.groupby(mode)['AVG_jaccard'].apply(list).to_dict()

# Get labels
labels_custom_bert = list(groups_custom_bert.keys())
labels_bert = list(groups_bert.keys())
labels_jaccard = list(groups_jaccard.keys())

# Save values to file
file = open(f"cohen_pc_{mode}.txt", "w", encoding="utf-8")

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

for i in range(len(labels_jaccard)):
    for j in range(i + 1, len(labels_jaccard)):
        label1 = labels_jaccard[i]
        label2 = labels_jaccard[j]
        # Cohens D
        cohen = cohenD(groups_jaccard[label1], groups_jaccard[label2])
        file.write(f"Compare Jaccard {label1} vs {label2}: Cohen's D={cohen:.3f}\n")

