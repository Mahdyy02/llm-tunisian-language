import pandas as pd

# Load CSV
df = pd.read_csv(fr"task 3\task3_classification_metrics.csv")

# Map LLM versions
versions = {
    "claude": "Claude Sonnet 4.5",
    "deepseek": "DeepSeek-R1",
    "gemini": "Gemini 2.5 Flash",
    "gpt": "GPT-4o Mini",
    "grok": "Grok 3",
    "mistral": "Mixtral 8×22B",
    "qwen": "Qwen 3"
}

# Columns to include in LaTeX
cols = [
    "accuracy", "f1_weighted", "f1_macro", "f1_positive", "f1_negative", "f1_neutral",
    "cohen_kappa", "mcc", "precision_positive", "precision_negative", "precision_neutral"
]

# Round all numeric values to 2 decimals
df[cols] = df[cols].round(2)

# Start LaTeX table
latex = r"""\begin{table*}[h!]
\centering
\resizebox{\textwidth}{!}{%
\begin{tabular}{|l|l|""" + "c|"*len(cols) + "}\n"
latex += r"\hline" + "\n"
latex += r"\textbf{LLM} & \textbf{Version (Nov 2025)} & " + " & ".join([f"\\textbf{{{c.replace('_',' ').title()}}}" for c in cols]) + r" \\ \hline" + "\n"

# Build table rows
for _, row in df.iterrows():
    llm = row['llm']
    version = versions.get(llm.lower(), "")
    values = []
    for c in cols:
        val = row[c]
        # Check if this is the max in the row (excluding LLM & Version)
        max_val = row[cols].max()
        if val == max_val:
            values.append(f"\\textbf{{{val}}}")
        else:
            values.append(f"{val}")
    latex += f"{llm.title()} & {version} & " + " & ".join(values) + r" \\ " + "\n"

latex += r"\hline" + "\n"
latex += r"\end{tabular}%" + "\n}\n"
latex += r"\caption{Sentiment classification performance of LLMs on the Tunisian dataset, including versions used (as of November 2025). Best values per row highlighted in bold.}" + "\n"
latex += r"\label{tab:task3_results}" + "\n"
latex += r"\end{table*}"

# Save LaTeX to file
with open("sentiment_table.tex", "w") as f:
    f.write(latex)

print("LaTeX table generated and saved to sentiment_table.tex")
