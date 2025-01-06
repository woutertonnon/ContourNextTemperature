import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import t

def fit_BG_curve(df):
    df['T-T_ref'] = df['T'] - df['T_ref']
    df['BG-BG_ref'] = df['BG'] - df['BG_ref']
    
    for var_y in ['BG-BG_ref']:
        var_x = 'T-T_ref'
        
        # Prepare the data
        X = np.array(df[var_x]).reshape(-1, 1)
        Y = np.array(df[var_y]).reshape(-1, 1)
        
        # Fit linear regression without intercept
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, Y)
        
        # Extract slope
        slope = reg.coef_[0][0]
        
        # Predict outputs
        x_plot = np.linspace(X.min()-10, X.max()+10, 100).reshape(-1, 1)
        y_pred = reg.predict(x_plot)
        
        # Calculate residuals and standard error
        residuals = Y - reg.predict(X)
        residual_sum_of_squares = np.sum(residuals**2)
        degrees_of_freedom = X.shape[0] - 1  # No intercept, so subtract 1
        variance = residual_sum_of_squares / degrees_of_freedom  # Residual variance
        standard_error = np.sqrt(variance)
        
        # Confidence interval calculation
        mean_X = np.mean(X)
        n = X.shape[0]
        t_value = t.ppf(1 - 0.025, degrees_of_freedom)  # 95% confidence (two-tailed)
        
        # Calculate confidence intervals
        confidence_interval = t_value * standard_error * np.sqrt(
            1 / n + ((x_plot - mean_X)**2 / np.sum((X - mean_X)**2))
        )
        
        # Upper and lower bounds
        lower_bound = y_pred - confidence_interval
        upper_bound = y_pred + confidence_interval
        
        # Plotting
        plt.figure()
        plt.scatter(X, Y, color='blue', label='Data Points')
        plt.plot(x_plot, y_pred, color='red', label=r"Linear Fit: $BG_{\text{diff}}$ = "+f"{slope:.4f}"+r'T_{\text{diff}}')
        plt.fill_between(x_plot.flatten(), lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.3, label='95% Confidence Interval')
        
        plt.grid(True)
        plt.xlabel(r"$T_{\text{diff}}$ [°C]")
        plt.ylabel(r'$BG_{\text{diff}}$ [mmol/L]')
        plt.xlim([-30,10])
        plt.legend()
        plt.savefig('BGdiffTdiff.png')

        plt.figure()
        reference_temperature = 22.5;
        absolute_temperature = x_plot+reference_temperature
        reported_5mmolL = y_pred+5
        lower_bound = reported_5mmolL - confidence_interval
        upper_bound = reported_5mmolL + confidence_interval
        plt.plot(absolute_temperature, reported_5mmolL, color='red', label=f'Reported Glucose for a real glucose of 5mmol/L')
        plt.fill_between(absolute_temperature.flatten(), lower_bound.flatten(), upper_bound.flatten(), color='gray', alpha=0.3, label='95% Confidence Interval')
        
        plt.grid(True)
        plt.xlabel(r"Temperature [°C]")
        plt.ylabel(r'Reported Blood Glucose [mmol/L]')
        plt.legend()
        plt.savefig("BGT.png")
        plt.show()



# Example Usage
df = pd.read_csv('data.csv') 
fit_BG_curve(df)
plt.show()
