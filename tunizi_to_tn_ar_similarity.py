import re
import Levenshtein

def normalize_arabic(text):
    """
    Normalize Arabic text for transliteration evaluation:
    - Remove diacritics (fat7a, damma, kasra, shadda, sukun)
    - Remove spaces
    - Optional: normalize Alef variations
    """
    diacritics = '[ًٌٍَُِّْ]'
    text = re.sub(diacritics, '', text)
    text = text.replace(' ', '')
    # Normalize Alef variants to standard Alef
    text = re.sub('[إأآا]', 'ا', text)
    return text

def cer(gt, ai):
    """
    Character Error Rate
    CER = (insertions + deletions + substitutions) / len(gt)
    """
    gt_norm = normalize_arabic(gt)
    ai_norm = normalize_arabic(ai)
    distance = Levenshtein.distance(gt_norm, ai_norm)
    return distance / len(gt_norm)

def levenshtein_distance(gt, ai):
    """
    Raw Levenshtein distance (number of edits)
    """
    gt_norm = normalize_arabic(gt)
    ai_norm = normalize_arabic(ai)
    return Levenshtein.distance(gt_norm, ai_norm)


def lcs_similarity(gt, ai):
    """
    LCS similarity: length of longest common subsequence / max length
    """
    gt_norm = normalize_arabic(gt)
    ai_norm = normalize_arabic(ai)

    m, n = len(gt_norm), len(ai_norm)
    # Create DP table
    L = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i==0 or j==0:
                L[i][j] = 0
            elif gt_norm[i-1] == ai_norm[j-1]:
                L[i][j] = L[i-1][j-1]+1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])
    lcs_len = L[m][n]
    # Normalize by maximum length
    return lcs_len / max(len(gt_norm), len(ai_norm))

