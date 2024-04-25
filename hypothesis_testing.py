import pandas as pd

# Replace 'path_to_your_file.xlsx' with the path to your Excel file
data = pd.read_excel(r'C:\Users\abhij\Masters\Spring 2024\Capstone_Project\.venv\output.xlsx')

# Assuming the columns are named 'Angles_Frontal' and 'Angles_NonFrontal'
angles_frontal = data['Angles_Frontal'].dropna()  # Remove any NaN values
angles_non_frontal = data['Angles_NonFrontal'].dropna()  # Remove any NaN values

from scipy.stats import shapiro

# Test normality for each group
stat_f, p_f = shapiro(angles_frontal)
stat_nf, p_nf = shapiro(angles_non_frontal)

print('Normality Test for Frontal: p-value =', p_f)
print('Normality Test for Non-Frontal: p-value =', p_nf)

from scipy.stats import mannwhitneyu

# Assuming you've already loaded your data into angles_frontal and angles_non_frontal
stat, p = mannwhitneyu(angles_frontal, angles_non_frontal, alternative='two-sided')

print('Mann-Whitney U Test Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution (fail to reject H0)')
else:
    print('Probably different distributions (reject H0)')

