import pandas as pd
df = pd.read_csv('sales_data_with_discounts.csv')

numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
mean_vals = df[numerical_cols].mean()
median_vals = df[numerical_cols].median()
mode_vals = df[numerical_cols].mode()
std_devs = df[numerical_cols].std()

# Data Visualization

import matplotlib.pyplot as plt
df[numerical_cols].hist(bins=20, figsize=(10,10))
plt.show()

#Box Plots
df[numerical_cols].plot(kind='box', subplots=True, layout=(2,3), figsize=(10,10))
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols].apply(lambda x: x.value_counts()).plot(kind='bar', figsize=(10,6))
plt.show()

#Standardization of Numerical Variables
standardized_df = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
standardized_df.hist(bins=20, figsize=(10,10))
plt.show()

#Conversion of Categorical Data into Dummy Variables
df_encoded = pd.get_dummies(df, columns=categorical_cols)
print(df_encoded.head())
