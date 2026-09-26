


#SETUP THE LIBRARIES



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from scipy.stats import f_oneway

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)

print("READY.\n\n")



#LOAD THE DATA



print("LOADING THE DATA:")
ab = pd.read_csv(r"C:\Users\kartik\Desktop\DSML PROJECT 2\ab_data.csv")
print("Dataset Shape:", ab.shape)
ab.head()



#INSPECT THE DATA



print("\n\nINSPECTION OF DATA:")
print("\nSHAPE OF DATASET:", ab.shape)
ab.head()
print("\nTAIL OF DATASET:", ab.tail())
print("\nINFORMATION OF DATASET:", ab.info())
print("\nDESCRIPTION OF DATASET:", ab.describe())
print("\nCOLUMNS OF DATASET:", ab.columns)



#CHECKING THE MISSING VALUES



print("\n\nTHE MISSING VALUES ARE:")
print(ab.isnull().sum())



#CHECK UNIQUE VALUES



print("\n\nTOTAL NUMBER OF UNIQUE VALUES")
print(ab["group"].value_counts())
print(ab["landing_page"].value_counts())
print(ab["converted"].value_counts())



#CHECKING THE NUMBER OF UNIQUE USERS



print("\n\nTHE NUMBER OF UNIQUE USERS")
print(ab["user_id"].nunique())
print(ab["user_id"].duplicated().sum())



#CLEANING OF DATA



#FIND THE MISMATCHED



print("\n\nMISMATCHED DATA:")
mismatch = (
    ((ab["group"] == "treatment") &
     (ab["landing_page"] != "new_page"))
    |
    ((ab["group"] == "control") &
     (ab["landing_page"] != "old_page"))
)



#REMOVE THE MISMATCHED DATA



ab_clean = ab[~mismatch]

print("\n\nShape After Removing Mismatches:")
print(ab_clean.shape)



#CHECKING THE DUPLICATES



print(
    "\n\nDuplicate Users:",
    ab_clean["user_id"].duplicated().sum()
)



#REMOVING THE DUPLICATES



print("\n\nMismatched Rows:", mismatch.sum())



#CLEANED DATA



print("\n\nCLEANED DATA:")
print(ab_clean.shape)



#VERIFY THE EXPERIMENT



print("\n\nVERIFICATION OF THE EXPERIMENT:")
pd.crosstab(
    ab_clean["group"],
    ab_clean["landing_page"]
)



#GROUP DISTRIBUTION



print("\n\nGROUP DISTRIBUTION:")
ab_clean["group"].value_counts()
plt.figure(figsize=(6,4))

sns.countplot(
    x="group",
    data=ab_clean
)

plt.title("Sample Size Per Group")
plt.show()



#CONVERSION RATE ANALYSIS



print("\n\nCONVERSION RATE ANALYSIS:")
conversion_rate = (
    ab_clean
    .groupby("group")["converted"]
    .mean()*100
)

conversion_rate



#CONVERSION RATE PLOT



print("\n\nCONVERSION RATE PLOT:")
plt.figure(figsize=(6,4))
sns.barplot(
    x="group",
    y="converted",
    data=ab_clean
)
plt.title("Conversion Rate by Group")
plt.ylabel("Conversion Rate")
plt.show()



#T-TEST



print("\n\nT-TEST:")
control = ab_clean[
    ab_clean["group"]=="control"
]["converted"]

treatment = ab_clean[
    ab_clean["group"]=="treatment"
]["converted"]

t_stat, p_value = ttest_ind(
    control,
    treatment,
    equal_var=False
)

print("T Statistic:", t_stat)
print("P Value:", p_value)



#CHI-SQUARE TEST



print("\n\nCHI-SQUARE TEST")
contingency = pd.crosstab(
    ab_clean["group"],
    ab_clean["converted"]
)
contingency

chi2, p, dof, expected = \
chi2_contingency(contingency)

print("Chi Square:", chi2)
print("P Value:", p)



#COUNTRY ANALYSIS



print("\n\nLOADING THE DATA OF COUNTRY")
countries = pd.read_csv(
    r"C:\Users\kartik\Desktop\DSML PROJECT 2\countries.csv"
)
df = ab_clean.merge(
    countries,
    on="user_id"
)



#COUNTRY-WISE CONVERSION



print("\n\nCOUNTRY-WISE CONVERSION")
country_conversion = (
    df.groupby("country")
      ["converted"]
      .mean()*100
)

country_conversion



#COUNTRY PLOT



print("\n\nCOUNTRY PLOTING")
plt.figure(figsize=(7,4))

sns.barplot(
    x="country",
    y="converted",
    data=df
)

plt.title("Conversion Rate by Country")

plt.show()



#ANNOVA TEST



print("\n\nANNOVA TEST")
us = df[df["country"]=="US"]["converted"]
uk = df[df["country"]=="UK"]["converted"]
ca = df[df["country"]=="CA"]["converted"]

f_stat, p_value = f_oneway(
    us,
    uk,
    ca
)

print("F Statistic:", f_stat)
print("P Value:", p_value)



#TEST COMPLETION



print("\n\nTEST COMPLETED")

