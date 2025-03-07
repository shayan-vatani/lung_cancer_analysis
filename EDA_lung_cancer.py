import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# 1️⃣ Exploratory Data Analysis (EDA) on Lung Cancer Dataset
# 📌 Objective: Understand the dataset structure, detect patterns, and visualize trends.
#
# ✅ Steps:
#
# Load the dataset using pandas and check its structure.
# Handle missing values and incorrect data.
# Identify data types (categorical vs. numerical) and encode categorical features.
# Use histograms, box plots, and violin plots to explore the distribution of numerical features.
# Generate correlation heatmaps to find relationships between features.
# 📊 Key Visualizations:
#
# Age distribution of patients.
# Gender breakdown of patients.
# Smoking history analysis.
# Correlation between features and lung cancer presence.

file=pd.read_csv('lung_cancer_prediction_dataset.csv')
df=file.dropna().reset_index(drop=True)
list=[]
list_cat=[]
for d in df.columns:
    if (df[d].dtypes==np.float64)|(df[d].dtypes==np.int64):
        list.append(d)
    else:
        list_cat.append(d)
list_n=[str(i) for i in list if i!='ID']

df[list_n].hist(bins=50,figsize=(10,5))
plt.tight_layout()
plt.show()
fig,axes=plt.subplots(len(list_cat)//2+1,2,figsize=(20,40))
axes=axes.flatten()
for i, col in enumerate(list_cat):
    counts=df[col].value_counts()
    counts.plot(kind='bar',ax=axes[i],title=f"distribution of {col}")
    axes[i].set_ylabel("count")
    axes[i].set_xlabel(col)
    axes[i].tick_params(axis='x',rotation=45)

for j in range(i+1,len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


correlation=df[list_n].corr()
print(correlation)
plt.figure(figsize=(20,20))

sns.heatmap(correlation, annot=True, cmap='YlGnBu',fmt=".2f",linewidths=.5)
plt.show()


sns.countplot(data=df,x="Gender",hue="Smoker")
plt.title('Gender vs Smoker')
plt.xlabel('Gender')
plt.ylabel('Smoker')
plt.show()