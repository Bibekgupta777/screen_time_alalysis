# ==============================================
# STEP 1: DATA LOADING & CLEANING
# ==============================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats# Paths
raw_file = r"C:\Desktop\Thesis_Projects\data\synthetic_screen_time_500k.csv"
save_path = r"C:\Desktop\Thesis_Projects\visualizations"
cleaned_file = r"C:\Desktop\Thesis_Projects\data\cleaned_screen_time.csv"
os.makedirs(save_path, exist_ok=True)# Load raw data
df = pd.read_csv(raw_file)

print(f"✅ Raw Data Shape: {df.shape}")
print("✅ Columns:", df.columns.tolist())
# ------------ CLEANING ------------
# ✅ Keep only Male & Female (Remove "Others")
df = df[df["gender"].isin(["Male", "Female"])]
# Drop duplicates
df.drop_duplicates(inplace=True)
# Drop rows with critical missing values
critical_cols = ["daily_screen_time_hours", "age", "sleep_duration_hours", "stress_level", "mental_health_score"]
df.dropna(subset=critical_cols, inplace=True)
# Remove impossible or extreme outliers (basic cleaning)
df = df[(df["daily_screen_time_hours"] > 0) & (df["daily_screen_time_hours"] <= 16)]
df = df[(df["age"] >= 10) & (df["age"] <= 80)]
df = df[(df["sleep_duration_hours"] > 0) & (df["sleep_duration_hours"] <= 15)]
# Fill remaining missing numeric values with median
for col in df.select_dtypes(include=np.number).columns:
    df[col].fillna(df[col].median(), inplace=True)
    # Derived Columns
df["screen_sleep_ratio"] = df["daily_screen_time_hours"] / df["sleep_duration_hours"]
df["total_device_usage"] = df[
    ["phone_usage_hours", "laptop_usage_hours", "tablet_usage_hours", "tv_usage_hours"]
].sum(axis=1)
df["screen_time_group"] = pd.cut(
    df["daily_screen_time_hours"], bins=[0,4,8,16], labels=["Low","Moderate","High"]
)
# Save cleaned data
df.to_csv(cleaned_file, index=False)
print(f"✅ Cleaned Data Shape (Male & Female Only): {df.shape}")
print(f"✅ Cleaned data saved at: {cleaned_file}")
# ==============================================
# STEP 2: HELPER FUNCTION TO SAVE FIGURES
# ==============================================

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)

def save_fig(name):
    plt.savefig(os.path.join(save_path, f"{name}.png"), dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {name}.png")
    # ==============================================
# STEP 3: DESCRIPTIVE VISUALIZATIONS (USING CLEANED DATA)
# ==============================================

# Histogram - Daily Screen Time
plt.figure(figsize=(7,5))
sns.histplot(df["daily_screen_time_hours"], bins=30, kde=True, color="coral")
plt.title("Distribution of Daily Screen Time (Male & Female Only)")
save_fig("hist_daily_screen_time")
plt.show()
# KDE - Age Distribution
plt.figure(figsize=(7,5))
sns.kdeplot(df["age"], fill=True, color="blue")
plt.title("Age Distribution of Participants (Male & Female Only)")
save_fig("kde_age_distribution")
plt.show()
# Boxplot - Gender vs Screen Time
plt.figure(figsize=(7,5))
sns.boxplot(x="gender", y="daily_screen_time_hours", data=df, palette="Set2")
plt.title("Gender-wise Screen Time Distribution")
save_fig("boxplot_gender_screen_time")
plt.show()
# Violin Plot - Location vs Stress Level
plt.figure(figsize=(7,5))
sns.violinplot(x="location_type", y="stress_level", data=df, palette="muted")
plt.title("Stress Level Distribution by Location")
save_fig("violin_location_stress")
plt.show()
# Countplot - Screen Time Groups by Gender
plt.figure(figsize=(6,5))
sns.countplot(x="screen_time_group", hue="gender", data=df, palette="Set2")
plt.title("Screen Time Groups by Gender")
save_fig("countplot_screen_groups_gender")
plt.show()
# Barplot - Avg Mental Health Score by Location
plt.figure(figsize=(7,5))
sns.barplot(x="location_type", y="mental_health_score", data=df, ci="sd", palette="muted")
plt.title("Average Mental Health Score by Location")
save_fig("barplot_mental_health_location")
plt.show()
# ==============================================
# STEP 4: RELATIONSHIP & CORRELATION VISUALIZATIONS
# ==============================================

# Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(),
            annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label':'Correlation'})
plt.title("Detailed Correlation Heatmap")
save_fig("heatmap_correlation")
plt.show()
# Pairplot - Key Behavioral Variables
sns.pairplot(df[["daily_screen_time_hours","sleep_duration_hours","stress_level","mental_health_score"]],
             diag_kind="kde", corner=True, plot_kws={'alpha':0.3})
plt.suptitle("Pairwise Relationships of Key Variables", y=1.02)
save_fig("pairplot_key_variables")
plt.show()
# Regression - Screen Time vs Mental Health
plt.figure(figsize=(7,5))
sns.regplot(x="daily_screen_time_hours", y="mental_health_score",
            data=df, scatter_kws={'alpha':0.2, 'color':'grey'},
            line_kws={'color':'red'})
plt.title("Regression: Screen Time vs Mental Health Score")
save_fig("regression_screen_mental_health")
plt.show()
# KDE - Sleep Duration by Screen Time Groups
plt.figure(figsize=(7,5))
sns.kdeplot(data=df, x="sleep_duration_hours", hue="screen_time_group", fill=True)
plt.title("Sleep Duration by Screen Time Groups")
save_fig("kde_sleep_screen_groups")
plt.show()
# ==============================================
# STEP 5: STATISTICAL TESTS (APA STYLE)
# ==============================================

male = df[df["gender"]=="Male"]["daily_screen_time_hours"]
female = df[df["gender"]=="Female"]["daily_screen_time_hours"]

t_stat, p_val = stats.ttest_ind(male, female, equal_var=False)
cohens_d = (male.mean() - female.mean()) / np.sqrt((male.std()**2 + female.std()**2) / 2)

anova_res = stats.f_oneway(
    df[df["location_type"]=="Urban"]["daily_screen_time_hours"],
    df[df["location_type"]=="Suburban"]["daily_screen_time_hours"],
    df[df["location_type"]=="Rural"]["daily_screen_time_hours"]
)
eta_sq = (anova_res.statistic * (len(df)-3)) / (
    anova_res.statistic*(len(df)-3) + (len(df)-3)
)

results_text = (
    "APA-STYLE STATISTICAL RESULTS (Male & Female Only):\n"
    f"T-test: t = {t_stat:.2f}, p = {p_val:.4f}, Cohen's d = {cohens_d:.2f}\n"
    f"ANOVA: F = {anova_res.statistic:.2f}, p = {anova_res.pvalue:.4f}, Eta² = {eta_sq:.2f}"
)

with open(os.path.join(save_path, "statistical_results.txt"), "w", encoding="utf-8") as f:
    f.write(results_text)

print(results_text)
# ==============================================
# STEP 6: SUMMARY TABLES (SAVED)
# ==============================================

# Top Correlated Variables Table
top_corr = df.select_dtypes(include=[np.number]).corr()["daily_screen_time_hours"]\
             .sort_values(key=abs, ascending=False)[1:6]
top_corr.to_csv(os.path.join(save_path, "table_top_correlations.csv"))

# Group-wise Mean Table (Gender x Screen Time Group)
group_table = df.groupby(["gender","screen_time_group"])[
    ["daily_screen_time_hours","stress_level","mental_health_score"]].mean().round(2)
group_table.to_csv(os.path.join(save_path, "table_groupwise_means.csv"))

print("\n✅ Tables saved as CSV in visualization folder")