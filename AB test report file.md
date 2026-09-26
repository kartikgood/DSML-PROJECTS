\# A/B Testing Analysis for an E-Commerce Website



\## Abstract



This project analyzes the performance of two landing pages using A/B Testing. The objective is to determine whether the new landing page improves conversion rates compared to the existing page. Data cleaning, conversion analysis, T-Test, and Chi-Square Test were performed to support the business decision.



\## Introduction



A/B Testing is a method used to compare two versions of a webpage. Users were divided into:



* \*\*Control Group\*\* – Existing page
* \*\*Treatment Group\*\* – New page



The goal was to determine whether the new page increases user conversions.



\## Data Cleaning



The dataset was cleaned by:



* Removing mismatched group and landing page records.
* Removing duplicate users.
* Verifying experiment consistency using cross-tabulation.



\## Exploratory Data Analysis



The distribution of users across groups was analyzed. Conversion rates for both groups were calculated and visualized using bar charts.



\## Statistical Testing



\### T-Test



Used to compare conversion rates between the control and treatment groups.



\### Chi-Square Test



Used to determine whether conversion depends on the landing page shown to users.



\## Results



* Conversion rates of both groups were compared.
* T-Test was performed to check statistical significance.
* Chi-Square Test was used to validate the findings.



The final conclusion depends on the generated p-values.



\## Tools Used



* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy



\## Conclusion



This project demonstrates how A/B Testing can be used to make data-driven business decisions. By applying statistical tests and analyzing conversion rates, organizations can evaluate whether a new webpage design provides a measurable improvement over the existing version.







