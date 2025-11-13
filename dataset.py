import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    cohen_kappa_score,
    matthews_corrcoef,
    classification_report
)
import numpy as np
from tunizi_to_tn_ar_similarity import cer, levenshtein_distance, lcs_similarity
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from bert_score import score
from pathlib import Path

df = pd.read_csv("dataset.csv", sep=';')

import nltk
nltk.download('punkt_tab')  # ← Add this line
nltk.download('wordnet')


###############################################################
######################### TASK 1 ##############################
###############################################################

# Path to the folder containing AI outputs
task1_folder = Path(r"task 1")

# Assuming you already have your functions cer(), levenshtein_distance(), lcs_similarity() defined
# and your ground truth dataframe df with column "tunisian_arabic"

results = []

# Iterate through all files in the folder
for file_path in task1_folder.iterdir():
    # Skip directories and files starting with "prompt_"
    if file_path.is_file() and not file_path.name.startswith("prompt_"):
        llm_name = file_path.stem.replace("task1_", "")  # Extract LLM name from filename
        
        # Read AI output lines
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Initialize sums
        cer_s = 0
        lev_s = 0
        lcs_s = 0
        
        # Compute metrics
        for i, ai in enumerate(lines):
            gt = df["tunisian_arabic"][i]
            cer_s += cer(gt, ai)
            lev_s += levenshtein_distance(gt, ai)
            lcs_s += lcs_similarity(gt, ai)
        
        n = len(lines)
        cer_avg = cer_s / n
        lev_avg = lev_s / n
        lcs_avg = lcs_s / n
        
        # Append results
        results.append({
            "llm": llm_name,
            "cer": cer_avg,
            "lev": lev_avg,
            "lcs": lcs_avg
        })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(task1_folder / "task1_metrics.csv", index=False)
print("Results saved to task1_metrics.csv")

################################################################
########################## TASK 2 ##############################
################################################################

task2_folder = Path(r"task 2")

results = []

# Iterate through all LLM output files
for file_path in task2_folder.iterdir():
    if file_path.is_file() and not file_path.name.startswith("prompt_"):
        llm_name = file_path.stem.replace("task2_", "")
        
        # Read AI output lines
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        
        # Initialize sums
        bleu_s = 0
        meteor_s = 0
        
        # Compute metrics
        for i, ai in enumerate(lines):
            if i >= len(df):
                print(f"Warning: {llm_name} has more lines than ground truth")
                break
                
            gt = df["english_translation"].iloc[i].strip()
            
            # Tokenize
            gt_tokens = word_tokenize(gt.lower())
            ai_tokens = word_tokenize(ai.lower())
            
            # BLEU
            bleu_s += sentence_bleu([gt_tokens], ai_tokens)
            
            # METEOR
            meteor_s += single_meteor_score(gt_tokens, ai_tokens)
        
        n = len(lines)
        bleu_avg = bleu_s / n
        meteor_avg = meteor_s / n
        
        # BERTScore
        gt_translations = df["english_translation"].iloc[:len(lines)].tolist()
        P, R, F1 = score(lines, gt_translations, lang="en", verbose=True)
        bert_avg = F1.mean().item()
        
        # Append results
        results.append({
            "llm": llm_name,
            "bleu": bleu_avg,
            "meteor": meteor_avg,
            "bertscore_f1": bert_avg
        })

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(task2_folder / "task2_translation_metrics.csv", index=False)
print("Results saved to task2_translation_metrics.csv")


###############################################################
######################### TASK 3 ##############################
################################################################

task3_folder = Path(r"task 3")

# Ground truth sentiment labels
# Assuming your CSV has a column like "sentiment" or "label"
# with values: "POSITIVE", "NEGATIVE", "NEUTRAL"

results = []

# Iterate through all LLM output files
for file_path in task3_folder.iterdir():
    if file_path.is_file() and not file_path.name.startswith("prompt_"):
        llm_name = file_path.stem.replace("task3_", "")
        
        # Read AI output lines
        with open(file_path, "r", encoding="utf-8") as f:
            predictions = [line.strip().upper() for line in f.readlines()]
        
        # Get ground truth labels (match the number of predictions)
        ground_truth = df["sentiment"].iloc[:len(predictions)].str.upper().tolist()
        
        # Make sure we have the same number of samples
        if len(predictions) != len(ground_truth):
            print(f"Warning: {llm_name} has {len(predictions)} predictions but {len(ground_truth)} ground truth labels")
            min_len = min(len(predictions), len(ground_truth))
            predictions = predictions[:min_len]
            ground_truth = ground_truth[:min_len]
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth, predictions)
        
        # Precision, Recall, F1 (weighted average across all classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth, 
            predictions, 
            average='weighted',
            zero_division=0
        )
        
        # Macro average (treats all classes equally)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            ground_truth, 
            predictions, 
            average='macro',
            zero_division=0
        )
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(ground_truth, predictions)
        
        # Matthews Correlation Coefficient
        # Note: MCC works best for binary, but can be used for multi-class
        # For multi-class, we'll calculate it but it might not be as interpretable
        try:
            mcc = matthews_corrcoef(ground_truth, predictions)
        except:
            mcc = np.nan
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
            ground_truth, 
            predictions, 
            average=None,
            labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth, predictions, labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'])
        
        # Append results
        result = {
            "llm": llm_name,
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "cohen_kappa": kappa,
            "mcc": mcc,
            # Per-class metrics
            "precision_positive": precision_per_class[0],
            "recall_positive": recall_per_class[0],
            "f1_positive": f1_per_class[0],
            "precision_negative": precision_per_class[1],
            "recall_negative": recall_per_class[1],
            "f1_negative": f1_per_class[1],
            "precision_neutral": precision_per_class[2],
            "recall_neutral": recall_per_class[2],
            "f1_neutral": f1_per_class[2],
        }
        
        results.append(result)
        
        # Print detailed report for this LLM
        print(f"\n{'='*60}")
        print(f"Results for: {llm_name}")
        print(f"{'='*60}")
        print(f"\nClassification Report:")
        print(classification_report(ground_truth, predictions, 
                                   labels=['POSITIVE', 'NEGATIVE', 'NEUTRAL'],
                                   zero_division=0))
        print(f"\nConfusion Matrix:")
        print(f"                  Predicted")
        print(f"               POS  NEG  NEU")
        print(f"Actual POS    {cm[0]}")
        print(f"       NEG    {cm[1]}")
        print(f"       NEU    {cm[2]}")

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv(task3_folder / "task3_classification_metrics.csv", index=False)
print(f"\n{'='*60}")
print("Results saved to task3_classification_metrics.csv")

print(f"{'='*60}")

