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
        X = np.array(df[var_x])
        Y = np.array(df[var_y])

        # We assume that we have the relation the data has the following form BGdiff_i = beta*Tdiff_i + epsilon_i,
        # where epsilon_i is i.i.d. and is normally distributed with mean 0 and unknown variance.
        
        # Compute slope (no-intercept linear regression)
        beta = np.sum(X*Y)/np.sum(X*X)

        # Predict outputs
        x_plot = np.linspace(X.min()-10, X.max()+10, 100).reshape(-1, 1)
        y_pred = beta*x_plot

        # Calculate residuals and standard error
        residuals = Y - beta*X
        residual_sum_of_squares = np.sum(residuals**2)
        degrees_of_freedom = X.shape[0] - 1  # No intercept, so subtract 1
        variance = residual_sum_of_squares / degrees_of_freedom  # Residual variance
        standard_error = np.sqrt(variance)/np.sqrt(np.sum(X*X))
        standard_error_pred = np.sqrt(variance/4.+x_plot*x_plot*variance/np.sum(X*X))
        print(np.sqrt(variance))
        # 95% regression confidence interval calculation
        t_value = t.ppf(1 - 0.025, degrees_of_freedom)  # 95% confidence (two-tailed)
        print(t_value)
        print("sigma^2 = " + str(variance/4))

        print('approximation beta: '+ str(beta))
        print('95\% confidence interval beta: '+ str([beta-t_value*standard_error,beta+t_value*standard_error]))
        print('approximation beta: '+ str(beta))

        regression_lower_bound = (beta-t_value*standard_error)*x_plot
        regression_upper_bound = (beta+t_value*standard_error)*x_plot
        prediction_lower_bound = 5 - beta*x_plot + t_value*standard_error_pred
        prediction_upper_bound = 5 - beta*x_plot - t_value*standard_error_pred
        print(beta)
        print(standard_error)
        print(beta+t_value*standard_error)
        print(x_plot*x_plot)

        # Plotting
        plt.figure()
        plt.scatter(X, Y, color='blue', label='Data Points')
        plt.plot(x_plot, y_pred, color='red', label=r"Linear Fit: $BG_{\text{diff}}$ = "+f"{beta:.4f}"+r'$T_{\text{diff}}$')
        plt.fill_between(x_plot.flatten(), regression_lower_bound.flatten(), regression_upper_bound.flatten(), color='gray', alpha=0.3, label='95% Regression Confidence Interval')
        
        plt.grid(True)
        plt.xlabel(r"$T_{\text{diff}}$ [°C]")
        plt.ylabel(r'$BG_{\text{diff}}$ [mmol/L]')
        plt.xlim([-30,10])
        plt.legend()
        plt.savefig('BGdiffTdiff.png',dpi=600)

        # Plotting
        plt.figure()
        plt.plot(x_plot+22.5, 5 - beta*x_plot, color='red', label=r"Maximum-likelihood: $BG_{\text{actual}}$ for $BG_{\text{Contour}} = 5$ mmol/L")
        plt.fill_between((x_plot+22.5).flatten(), prediction_lower_bound.flatten(), prediction_upper_bound.flatten(), color='gray', alpha=0.3, label='95% Prediction Confidence Interval')
        
        plt.grid(True)
        plt.xlabel(r"$T$ [°C]")
        plt.ylabel(r'$BG_{\text{actual}}$ [mmol/L]')
        plt.xlim([-30+22.5,10+22.5])
        plt.legend()
        plt.savefig('BGT.png',dpi=600)


        


# Example Usage
df = pd.read_csv('data.csv') 
fit_BG_curve(df)
plt.show()
