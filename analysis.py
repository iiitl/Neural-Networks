import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# 1. Plot Multi-class Quality Score
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='quality', palette='viridis')
plt.title('Distribution of Multi-class Quality Score')
plt.savefig('multi_class_plot.png')
plt.close()

# 2. Plot Binary Quality Target
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='quality_binary', palette='coolwarm')
plt.title('Distribution of Binary Quality Target')
plt.savefig('binary_class_plot.png')
plt.close()

print("Plots generated successfully: multi_class_plot.png and binary_class_plot.png")

# Quick check on class imbalance for your insights
print("\nMulti-class distribution (%):")
print(df['quality'].value_counts(normalize=True) * 100)
print("\nBinary distribution (%):")
print(df['quality_binary'].value_counts(normalize=True) * 100)
