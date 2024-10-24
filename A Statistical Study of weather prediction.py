#!/usr/bin/env python
# coding: utf-8

# # A Statistical study of weather prediction

# In[1]:


# import the libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.preprocessing import RobustScaler


# In[2]:


# Load the dataset
df = pd.read_excel('W Data.xlsx')
df


# In[3]:


# it give information about the data
df.info()


# In[4]:


# check the missing values
df.isnull().sum()


# In[5]:


# check duplicated values
df[df.duplicated()]


# In[6]:


# drop null values
df.dropna()


# # Exploratory Data Analysis

# In[7]:


# only numerical columns
numerical_columns = ['temperature_celsius', 'wind_kph', 'wind_degree', 'pressure_in', 
                     'humidity', 'cloud', 'feels_like_celsius', 'visibility_km']


# In[8]:


# Box plot : Detect the outliers
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_columns])
plt.title("Box Plot")
plt.show()


# In[9]:


# Box plots 
for column in numerical_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(f'Box plot of {column}')
    plt.xlabel(column)
    plt.show()


# # Descriptive statistics

# In[10]:


# Assuming your DataFrame is named 'df'
# You can use df.describe() for numerical columns and df[column].value_counts() for categorical columns
  # Summary statistics for numerical columns
df.describe()


# # Check the distribution

# In[11]:



# Histograms for numerical columns
numerical_columns = ['temperature_celsius', 'wind_kph', 'pressure_in', 
                     'humidity', 'cloud', 'feels_like_celsius', 'visibility_km']
df[numerical_columns].hist(bins=20, figsize=(15, 10))
plt.suptitle('Histograms of Numerical Columns')
plt.show()


# In[12]:


# Bar plots for categorical columns
categorical_columns = ['condition_text']
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.show()


# #### Interpretation :
#     From the above figure we can observe that the condition text is high in mist condition.

# In[13]:


# Bar plots for categorical columns
categorical_columns = ['wind_direction']
for column in categorical_columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=column)
    plt.title(f'Countplot of {column}')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    plt.show()


# #### Interpretation :
#     From the above figure we can observe that the wind direction is high in North direction.

# # Correlation Heatmap

# In[14]:


# Correlation matrix and heatmap for numerical columns
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Numerical Columns')
plt.show()


# #### Interpretation :
#     From the above heatmap results we can see temperature_celsius is highly correlated with feels_like_celsius.

# ## Line graph

# In[15]:


# Assuming 'last_updated' is a timestamp column
df['last_updated'] = pd.to_datetime(df['last_updated'])
plt.figure(figsize=(12, 6))
plt.plot(df['last_updated'], df['temperature_celsius'], linestyle='-')
plt.title('Temperature Variation Over Time')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()


# #### Interpretation :
#     From the above figure we can observe that the tempuratue is decrease.

# # Ordinal Logistic regression
#     Ordinal logistic regression is used to model the relationship between an ordered multilevel dependent variable and independent variables. In the modeling, values of the dependent variable have a natural order or ranking. 

# In[16]:


pip install mord


# #### Coding on condition text column 
#     Clear = 0
#     Fog = 1
#     mist =2 
#     Moderate or heavy rain with thunder = 3
#     overcast = 4
#     partly cloudy = 5

#     Here we use variable selection method for better result, we check which column condition text is related to this column by using R programming. Then we got 4 columns 'visibility_km', 'pressure_in', 'cloud', 'wind_kph'.

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split
from mord import LogisticAT
from sklearn.metrics import accuracy_score

# Load your dataset (assuming it's already loaded into weather_data DataFrame)
weather_data = pd.read_excel('coding(W data).xlsx')

# Define independent variables (X) and dependent variable (y)
X = weather_data[['visibility_km', 'pressure_in', 'cloud', 'wind_kph']]
y = weather_data['condition_text']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the ordinal logistic regression model
model = LogisticAT()

# Train the model
model.fit(X_train, y_train)

# Predict the target variable
y_pred = model.predict(X_test)

# Calculate accuracy
acc1 = accuracy_score(y_test, y_pred)
print("testing Accuracy:", acc1)


# #### Interpretation :
#     Accuracy of ordinal logistic regression is 0.8%.

# In[18]:


from sklearn.metrics import confusion_matrix, classification_report

# Draw confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate evaluation measures
eval_report = classification_report(y_test, y_pred)
print("Evaluation Measures:")
print(eval_report)


# In[19]:


import pandas as pd
from mord import LogisticAT

# Load your trained model (assuming it's already trained and saved)
# model = LogisticAT.load('your_model_path.pkl')

# Prepare input data for prediction
new_data = pd.DataFrame({
    'visibility_km': [10.5],  # Example visibility in kilometers
    'pressure_in': [29.5],      # Example pressure in inches
    'cloud': [3],               # Example cloud cover level
    'wind_kph': [15.0]          # Example wind speed in kilometers per hour
})

# Make predictions
predictions = model.predict(new_data)

# Print predictions
print("Predicted weather condition:", predictions)


# #### Interpretation :
#     From the above code, if we enter the values of visibility_km, pressure_in, cloud, wind_kph in our mind, then this model give us the weather condition.
#     i.e.  Moderate or heavy rain with thunder = 3

# # ARIMA model

# In[20]:


#Import required Libraries
import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from numpy import log
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from pandas import read_csv
import multiprocessing as mp

### Just to remove warnings to prettify the notebook. 
import warnings
warnings.filterwarnings("ignore")


# In[21]:


# Load the dataset
df = pd.read_excel('W Data.xlsx')
df.tail()


# In[22]:


# Visualize the data
df['temperature_celsius'].plot(figsize=(12, 6), title='Temperature Time Series')


# In[23]:


# Set 'last_updated' as the index
df.set_index('last_updated', inplace=True)


# In[24]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Assuming 'df' is your DataFrame with time series data and 'last_updated' is already in datetime format

# Define a function for stationarity check
def check_stationarity(timeseries):
    # Calculate rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # Plot rolling statistics
    plt.figure(figsize=(10, 6))
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolling_mean, color='red', label='Rolling Mean')
    plt.plot(rolling_std, color='green', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    # Perform Augmented Dickey-Fuller test
    adf_test = adfuller(timeseries, autolag='AIC')
    adf_results = pd.Series(adf_test[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    print('Augmented Dickey-Fuller Test:')
    print(adf_results)
    for key, value in adf_test[4].items():
        adf_results['Critical Value (%s)' %key] = value
    print(adf_results)

# Perform stationarity check on 'temperature_celsius' column
check_stationarity(df['temperature_celsius'])


# In[25]:


df['temperature_diff'] = df['temperature_celsius'].diff()

df.dropna(inplace =True)

#plot differenced time series
plt.figure(figsize = (10,6))
plt.plot(df['temperature_diff'],color = 'blue')
plt.title('First order differenced Temperature Time Series')
plt.xlabel('Date')
plt.ylabel('Temperature Difference')
plt.show()
# perform stationarity check on differenced time series
check_stationarity(df['temperature_diff'])


# In[26]:


# Visualize the data
df['temperature_diff'].plot(figsize=(12, 6), title='Differencing Temperature Time Series')


# In[27]:


# Filter data within the specified date range
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

start_date = '2023-08-29'
end_date = '2023-12-07'
df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]

# Plot ACF
plt.figure(figsize=(12, 6))
plot_acf(df_filtered['temperature_diff'],lags= 15 , ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')
plt.xlabel('Lag')
plt.ylabel('ACF')
plt.show()


# In[28]:


# Filter data within the specified date range
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
start_date = '2023-08-29'
end_date = '2023-12-07'
df_filtered = df[(df.index >= start_date) & (df.index <= end_date)]
# Plot ACF
plt.figure(figsize=(12, 6))
plot_pacf(df_filtered['temperature_diff'],lags= 15 , ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.show()


# In[29]:


import statsmodels.api as sm
# Fit ARIMA model
arima_model = sm.tsa.ARIMA(df['temperature_diff'], order=(4,1,6))
arima_fit = arima_model.fit()
# Print summary of the fitted model
print(arima_fit.summary())


# In[30]:


# Fit ARIMA model
arima_model = sm.tsa.ARIMA(df['temperature_diff'], order=(5,1,6))
arima_fit = arima_model.fit()

print(arima_fit.summary())


# ## Best Fit(4,1,0)

# In[31]:


import statsmodels.api as sm
# Fit ARIMA model
arima_model = sm.tsa.ARIMA(df['temperature_diff'], order=(4,1,0))  
arima_fit = arima_model.fit()
print(arima_fit.summary())


# In[32]:


# Fit ARIMA model
arima_model = sm.tsa.ARIMA(df['temperature_diff'], order=(5,1,0))  
arima_fit = arima_model.fit()
print(arima_fit.summary())


# In[33]:


best_params = (4,1,0)


# In[34]:


# Fit ARIMA model
arima_model = sm.tsa.ARIMA(df['temperature_diff'], order=best_params)  
arima_fit = arima_model.fit()
print(arima_fit.summary())


# In[35]:


import matplotlib.pyplot as plt
# Get the residuals from the ARIMA model
residuals = arima_fit.resid
# Plot the residuals
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.xlabel('Index or Time')
plt.ylabel('Residuals')
plt.grid(True)
plt.show()


# In[36]:


from pandas import datetime
from pandas import read_csv
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot
# line plot of residuals
residuals = pd.DataFrame(arima_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()


# In[37]:


# 1. Model Diagnostics
arima_fit.plot_diagnostics(figsize=(10, 8))
plt.show()


# In[38]:


# summary stats of residuals
print(residuals.describe())


# In[39]:


import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Assuming 'df' is your DataFrame containing the time series data

# Define the size of the training set
train_size = int(len(df) * 0.8)  # 80% of data for training

# Initialize lists to store forecasted values
history = list(df[:train_size]['temperature_diff'])
predictions = []

# Iterate through the test set
for t in range(train_size, len(df)):
    # Fit ARIMA model
    model = ARIMA(history, order=best_params)  # Replace (p, d, q) with appropriate values
    model_fit = model.fit()

    # Forecast next value
    forecast = model_fit.forecast()[0]

    # Print forecasted and actual values
    print(f"Forecast={forecast:.3f}, Actual={df.iloc[t]['temperature_diff']:.0f}")

    # Append forecast to predictions list
    predictions.append(forecast)

    # Update history with actual value from test set
    history.append(df.iloc[t]['temperature_diff'])

# Calculate and print RMSE
rmse = np.sqrt(mean_squared_error(df[train_size:]['temperature_diff'], predictions))
print(f"Root Mean Squared Error (RMSE): {rmse}")


# In[40]:


# Plot actual vs. predicted values
plt.plot(df[train_size:].index, df[train_size:]['temperature_diff'], label='Actual')
plt.plot(df[train_size:].index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Values')
plt.xticks(rotation=45)
plt.legend()
plt.show()


# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Fit ARIMA model
model = ARIMA(df['temperature_diff'], order=best_params)
arima_fit = model.fit()

# Forecast future difference values
forecast_steps = 11  # Number of time steps to forecast into the future
forecast_diff = arima_fit.forecast(steps=forecast_steps)

# Convert forecasted difference values back to original scale
last_observed_value = df['temperature_celsius'].iloc[-1]  # Last observed value from the original series
forecast_original = np.cumsum(forecast_diff) + last_observed_value
print("forecast values :",forecast_original)


# In[42]:


# Calculate mean and standard deviation of temperature data
mean_temp = df['temperature_celsius'].mean()
std_dev = df['temperature_celsius'].std()

# Convert confidence interval bounds to original scale
forecast = arima_fit.get_forecast(steps=forecast_steps)
forecast_conf_int = forecast.conf_int()
forecast_conf_int_original = forecast_conf_int * std_dev + mean_temp

# Plot forecasted values with 95% confidence interval in original scale
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['temperature_celsius'], label='Observed', color='blue')  # Plot observed values
plt.plot(pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast_original, label='Forecast', color='red')  # Plot forecasted values
plt.fill_between(pd.date_range(start=df.index[-1], periods=forecast_steps+1, freq='D')[1:], forecast_conf_int_original.iloc[:, 0], forecast_conf_int_original.iloc[:, 1], color='grey', alpha=0.3, label='95% Confidence Interval')  # Plot 95% confidence interval
plt.xlabel('Time')
plt.ylabel('Temperature (Celsius)')
plt.title('Forecast using ARIMA Model')
plt.legend()
plt.xticks(rotation=45)
plt.show()


# In[ ]:




