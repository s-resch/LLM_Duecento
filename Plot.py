import pandas as pd
import matplotlib.pyplot as plt

type = "Summary"
csv_type = ""

if type == "Summary":
    csv_type = "Rias"
else:
    csv_type = "PC"

df = pd.read_csv(f'metrics_csv/Metriken_{csv_type}_Cleaned.csv',
    delimiter=";",
    decimal=",",
    encoding="utf-8")

mode = "text"
# mode = "model"

metrics = "AVG_BERT_custom_f1"
metrics_label = "BERTScore (Italian) F1-Score"

x_datapoints = df[mode].unique()
data = [df[df[mode] == x_datapoint][metrics].dropna() for x_datapoint in x_datapoints]

# Configure boxplot
fig, ax = plt.subplots(figsize=(10, 7))
ax.set(
    axisbelow=True,
    title=f'{type} {metrics_label} - {type}',
    ylabel=metrics_label,
)
ax.set_ylim(0.0, 1.0)
labels = []
# Abbreviate labels so we can display them properly
for m in x_datapoints:
    if "cogito" in m:
        m = "cogito"
    elif m == "mistral":
        m = "mist.7b"
    elif "mistral-large" in m:
        m = "mist. large"
    elif "openai" in m:
        m = "openai"
    elif m == "dante":
        m = "dante"
    elif "teuken" in m:
        m = "teuken"
    elif "70" in m:
        m = "llama70b"
    elif "meta" in m:
        m = "llama8b"
    elif "mini" in m:
        m = "minstr"
    elif "gemma" in m:
        m = "gemma"

    elif "Pacino" in m:
        m = "Pacino"
    elif "Papafava" in m:
        m = "Fr. Pap."
    elif "Danza" in m:
        m = "Danza M."
    elif "Paganino" in m:
        m = "Paganino"
    elif "Tommaso" in m:
        m = "Tommaso"
    elif "Alberto" in m:
        m = "Alberto"
    elif "Cavalcanti" in m:
        m = "Cavalcanti"
    elif "Pascoli" in m:
        m = "Pascoli"

    labels.append(m)

ax.set_xticklabels(labels, rotation=45, fontsize=7)
bp = ax.boxplot(data, patch_artist=True)
for box in bp["boxes"]:
    box.set_facecolor("white")
for median in bp['medians']:
    median.set_color('black')

plt.savefig(rf"boxplots/Print_Boxplot_{metrics}_{type}_{mode}.png", dpi=300, bbox_inches="tight")