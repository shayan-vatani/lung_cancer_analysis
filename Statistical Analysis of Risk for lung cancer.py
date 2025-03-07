import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
"""
2️⃣ Statistical Analysis of Risk Factors for Lung Cancer
📌 Objective: Identify significant risk factors for lung cancer using statistical tests.

✅ Steps:

Use chi-square tests to analyze categorical variables like smoking status, gender, and cancer presence.
Use t-tests or ANOVA to compare numerical variables (like age) across different cancer statuses.
Interpret p-values to determine which factors are statistically significant.
Create bar plots to visualize categorical relationships.
Generate statistical summaries for lung cancer vs. non-lung cancer groups.
📊 Key Insights:

Are smokers significantly more likely to have lung cancer?
Does age impact lung cancer prevalence?
Are there gender-based differences in lung cancer cases?
"""

df=pd.read_csv('lung_cancer_prediction_dataset.csv')
print(df['Lung_Cancer_Diagnosis'].value_counts())
contingency_table=pd.crosstab(df['Smoker'],df['Lung_Cancer_Diagnosis'])
chi2, p_value, dof, expected=stats.chi2_contingency(contingency_table)
print(f"the chi-square test statistic is {chi2}")
print(f"the p-value is {p_value}")
print(f"the dof is {dof}")
print(f"the expected value is {expected}")
contingency_table.plot(kind='bar',figsize=(15,10))
plt.title('Contingency Table')
plt.show()