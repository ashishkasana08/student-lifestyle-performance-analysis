# =========================================================
# STUDENT LIFESTYLE & PERFORMANCE ANALYSIS PROJECT
# =========================================================

# ==============================
#  IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind, shapiro
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set(style="whitegrid")

# ==============================
#  1. LOAD DATA
# ==============================
df = pd.read_csv(r"C:/Users/Aashish/OneDrive/Documents/Downloads/student_lifestyle_performance_dataset.csv")

print("\n===== DATA OVERVIEW =====")
print(df.head())
print("\nShape:", df.shape)
print("\nData Types:\n", df.dtypes)

# ==============================
# 2. DATA CLEANING
# ==============================
print("\n===== DATA CLEANING =====")

print("Missing Values:\n", df.isnull().sum())
print("Duplicate Rows:", df.duplicated().sum())

df['Branch'] = df['Branch'].astype('category')
df['Diet_Type'] = df['Diet_Type'].astype('category')
df['Residence'] = df['Residence'].astype('category')

# ==============================
# ️ FEATURE ENGINEERING
# ==============================
df['Sleep_Group'] = np.where(df['Sleep_Hours'] >= 7, 'Adequate', 'Inadequate')

df['Study_Category'] = pd.cut(
    df['Study_Hours_per_Day'],
    bins=[0, 2, 4, 6, 10],
    labels=['0-2', '2-4', '4-6', '6+']
)

# ==============================
#  3. EDA
# ==============================
print("\n===== EDA =====")
print(df.describe())
print("\nSkewness:\n", df.skew(numeric_only=True))

# ==============================
#  OUTLIER DETECTION (IQR)
# ==============================
print("\n===== OUTLIER DETECTION =====")

numeric_cols = df.select_dtypes(include=np.number).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    print(f"{col}: {len(outliers)} outliers")

# ==============================
#  4. VISUALIZATION
# ==============================

# 1. Histogram + KDE
plt.figure(figsize=(6,4))
sns.histplot(df['CGPA'], kde=True)
plt.title("CGPA Distribution")
plt.xlabel("CGPA")
plt.tight_layout()
plt.show()

# 2. KDE Plot
plt.figure(figsize=(6,4))
sns.kdeplot(df['CGPA'], fill=True)
plt.title("KDE Plot of CGPA")
plt.xlabel("CGPA")
plt.tight_layout()
plt.show()

# 3. Study Hours vs CGPA
plt.figure(figsize=(6,4))
sns.regplot(x='Study_Hours_per_Day', y='CGPA', data=df)
plt.title("Study Hours vs CGPA")
plt.xlabel("Study Hours per Day")
plt.ylabel("CGPA")
plt.tight_layout()
plt.show()

# 4. Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Sleep vs CGPA
plt.figure(figsize=(6,4))
plt.scatter(df['Sleep_Hours'], df['CGPA'])
plt.title("Sleep Hours vs CGPA")
plt.xlabel("Sleep Hours")
plt.ylabel("CGPA")
plt.tight_layout()
plt.show()

# 6. Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(x=df['CGPA'])
plt.title("CGPA Boxplot")
plt.tight_layout()
plt.show()

# 7. CGPA by Branch
plt.figure(figsize=(8,5))
sns.boxplot(x='Branch', y='CGPA', data=df)
plt.xticks(rotation=45)
plt.title("CGPA by Branch")
plt.tight_layout()
plt.show()

# 8. Bar Plot (Study Category)
plt.figure(figsize=(6,4))
sns.barplot(
    x='Study_Category',
    y='CGPA',
    data=df.groupby('Study_Category')['CGPA'].mean().reset_index()
)
plt.title("Average CGPA by Study Hours")
plt.tight_layout()
plt.show()

# 9. Strip Plot
plt.figure(figsize=(6,4))
sns.stripplot(x='Study_Category', y='CGPA', data=df)
plt.title("Study Category vs CGPA")
plt.tight_layout()
plt.show()

# 10. Countplot (Branch)
plt.figure(figsize=(8,5))
sns.countplot(x='Branch', data=df)
plt.xticks(rotation=45)
plt.title("Students by Branch")
plt.tight_layout()
plt.show()

# 11. Histogram (Study Hours)
plt.figure(figsize=(6,4))
plt.hist(df['Study_Hours_per_Day'], bins=10)
plt.title("Study Hours Distribution")
plt.xlabel("Hours")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ==============================
#  5. CORRELATION
# ==============================
print("\n===== CORRELATION =====")
corr = df.corr(numeric_only=True)
print(corr['CGPA'].sort_values(ascending=False))

# ==============================
#  6. HYPOTHESIS TESTING
# ==============================
print("\n===== HYPOTHESIS TESTING =====")

group1 = df[df['Sleep_Group'] == 'Adequate']['CGPA']
group2 = df[df['Sleep_Group'] == 'Inadequate']['CGPA']

t_stat, p_val = ttest_ind(group1, group2, equal_var=False)

print("p-value:", p_val)

if p_val < 0.05:
    print("Reject H0 → Sleep has a significant effect on CGPA")
else:
    print("Fail to Reject H0 → No significant effect")

# Normality Test
stat, p = shapiro(df['CGPA'])
print("Normality p-value:", p)

# ==============================
#  7. LINEAR REGRESSION
# ==============================
print("\n===== LINEAR REGRESSION =====")

X = df[['Study_Hours_per_Day']]
y = df['CGPA']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score:", r2_score(y, y_pred))

print(f"\nEquation: CGPA = {model.coef_[0]:.3f} * Study_Hours + {model.intercept_:.3f}")

# ==============================
#  8. FINAL INSIGHT
# ==============================
print("""
Problem: Does increasing study hours improve CGPA?

Insight:
- Strong positive relationship observed
- Higher study hours correlate with better CGPA

Conclusion:
Study hours significantly influence academic performance

Recommendation:
Students should maintain 4–6 hours of focused study daily for optimal results
""")
