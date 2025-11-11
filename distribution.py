import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset CSV
df = pd.read_csv(r"dataset.csv", sep=";")

# Check the first few rows
print(df.head())

# Count the number of comments in each sentiment class
sentiment_counts = df['sentiment'].value_counts()
print("Sentiment counts:")
print(sentiment_counts)

# Calculate percentages
sentiment_percent = (sentiment_counts / len(df) * 100).round(2)
print("\nSentiment percentages (%):")
print(sentiment_percent)

# Plot histogram with larger text
plt.figure(figsize=(8,6))  # Increased figure size
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
plt.title('Distribution of Sentiment Classes', fontsize=18, fontweight='bold')  # Larger title
plt.xlabel('Sentiment', fontsize=16)  # Larger x-label
plt.ylabel('Number of Comments', fontsize=16)  # Larger y-label
plt.xticks(fontsize=14)  # Larger x-tick labels
plt.yticks(fontsize=14)  # Larger y-tick labels

# Larger text on bars
for i, val in enumerate(sentiment_counts.values):
    plt.text(i, val+0.5, f'{val} ({sentiment_percent[i]}%)', 
             ha='center', fontsize=14, fontweight='bold')  # Increased from 10 to 14

plt.tight_layout()
plt.show()