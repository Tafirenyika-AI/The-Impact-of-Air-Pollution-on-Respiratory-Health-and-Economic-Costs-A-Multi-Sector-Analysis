# =============================================================================
# Streamlit‑ready version (minimal edits)
# =============================================================================
# Import necessary libraries
import streamlit as st                       # NEW
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Air‑Pollution & Health‑Spend Analysis",
                   layout="wide")            # NEW
st.title("Air Pollution vs Health Expenditure")

# ----------------------------
# LOAD DATASETS
# ----------------------------
pollution_df = pd.read_csv(r"https://github.com/Tafirenyika-AI/The-Impact-of-Air-Pollution-on-Respiratory-Health-and-Economic-Costs-A-Multi-Sector-Analysis/blob/main/airq2.csv")
economic_df  = pd.read_csv(r"https://github.com/Tafirenyika-AI/The-Impact-of-Air-Pollution-on-Respiratory-Health-and-Economic-Costs-A-Multi-Sector-Analysis/blob/main/data1.csv")
health_df    = pd.read_csv(r"https://github.com/Tafirenyika-AI/The-Impact-of-Air-Pollution-on-Respiratory-Health-and-Economic-Costs-A-Multi-Sector-Analysis/blob/main/res.csv")

# ----------------------------
# CLEANING DATASETS
# ----------------------------
pollution_df = pollution_df[['Country', 'PM2.5 AQI Value']].dropna()
pollution_df = pollution_df.groupby('Country').mean().reset_index()

years = ["2017 [YR2017]", "2018 [YR2018]", "2019 [YR2019]", "2020 [YR2020]", "2021 [YR2021]"]
economic_df = economic_df[["Country Name"] + years]
economic_df.columns = ["Country"] + [y[:4] for y in years]
economic_df.replace("..", np.nan, inplace=True)
economic_df.iloc[:, 1:] = economic_df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')

health_df['season'] = health_df['season'].astype(str)
health_summary = health_df.groupby("season")["percent"].mean().reset_index()
health_summary.columns = ["Year", "Avg_Respiratory_Visit_Percent"]
health_summary["Year"] = health_summary["Year"].str[:4]
health_summary = health_summary.groupby("Year").mean().reset_index()
health_summary = health_summary[health_summary['Year'].str.isnumeric()]
health_summary['Year'] = health_summary['Year'].astype(int)

# ----------------------------
# MERGING DATASETS
# ----------------------------
merged_df = pd.merge(pollution_df, economic_df, on="Country", how="inner")
merged_df['Avg_Health_Expenditure'] = merged_df[['2017', '2018', '2019', '2020', '2021']].mean(axis=1)
merged_df = merged_df[['Country', 'PM2.5 AQI Value', 'Avg_Health_Expenditure']].dropna()
merged_df.to_csv(r"C:\Users\tafis\OneDrive\Desktop\Air_pollution\merged_clean_data.csv", index=False)

st.success("Merged clean dataset saved as 'merged_clean_data.csv'")

# ----------------------------
# ANALYSIS & CORRELATION
# ----------------------------
st.subheader("Correlation Analysis")
correlation = merged_df[['PM2.5 AQI Value', 'Avg_Health_Expenditure']].corr()
st.write(correlation)

# ----------------------------
# REGRESSION MODEL
# ----------------------------
st.subheader("Regression Analysis")
X = merged_df[['PM2.5 AQI Value']]
y = merged_df['Avg_Health_Expenditure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write(f"**Coefficient:** {model.coef_[0]:.4f}")
st.write(f"**Intercept:** {model.intercept_:.4f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.4f}")
st.write(f"**RMSE:** {mean_squared_error(y_test, y_pred, squared=False):.4f}")

# ----------------------------
# VISUALIZATION
# ----------------------------
# 1. PM2.5 AQI vs Health Expenditure
fig1 = plt.figure(figsize=(10, 6))
sns.scatterplot(x='PM2.5 AQI Value', y='Avg_Health_Expenditure', data=merged_df)
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.title('Figure 1: PM2.5 AQI vs Health Expenditure (% of GDP)')
plt.xlabel('Average PM2.5 AQI (Air Quality Index)')
plt.ylabel('Health Expenditure (% of GDP)')
plt.grid(True)
plt.legend(["Data Points", "Regression Line"])
st.pyplot(fig1)                              # NEW

# 2. Trend of respiratory admissions over time
fig2 = plt.figure(figsize=(10, 6))
sns.lineplot(data=health_summary, x='Year', y='Avg_Respiratory_Visit_Percent', marker='o')
plt.title('Figure 2: Trend of Respiratory Admissions Over Time')
plt.xlabel('Year')
plt.ylabel('Avg % of Respiratory‑Related Visits')
plt.grid(True)
plt.legend(["Avg Respiratory Visits (%)"])
st.pyplot(fig2)                              # NEW

# 3. Health Expenditure Over Time
expenditure_over_time = economic_df.drop(columns=['Country']).set_index(economic_df['Country']).dropna()
yearly_avg = expenditure_over_time.mean().reset_index()
yearly_avg.columns = ['Year', 'Avg_Health_Expenditure']
yearly_avg['Year'] = yearly_avg['Year'].astype(str)

fig3 = plt.figure(figsize=(10, 6))
sns.lineplot(data=yearly_avg, x='Year', y='Avg_Health_Expenditure', marker='o')
plt.title('Figure 3: Global Avg Health Expenditure Over Time (% of GDP)')
plt.xlabel('Year')
plt.ylabel('Avg Health Expenditure (% of GDP)')
plt.grid(True)
plt.legend(["Avg Global Expenditure"])
st.pyplot(fig3)                              # NEW

# ----------------------------
# FUTURE COST PREDICTION
# ----------------------------
st.subheader("Future Cost Prediction")
future_PM25 = pd.DataFrame({'PM2.5 AQI Value': np.linspace(10, 150, 30)})
future_expenditure = model.predict(future_PM25)

pred_df = pd.DataFrame({
    'PM2.5 AQI Value': future_PM25['PM2.5 AQI Value'],
    'Predicted Expenditure (% GDP)': future_expenditure
})
st.dataframe(pred_df)

fig4 = plt.figure(figsize=(10, 6))
plt.plot(future_PM25, future_expenditure, marker='o')
plt.title('Figure 4: Predicted Health Expenditure vs PM2.5 AQI')
plt.xlabel('PM2.5 AQI Value')
plt.ylabel('Predicted Health Expenditure (% of GDP)')
plt.grid(True)
plt.legend(["Predicted Trend"])
st.pyplot(fig4)                              # NEW
