import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

def fit_BG_curve(df):
    # Compute differences relative to reference values
    df['T_diff'] = df['T'] - df['T_ref']
    df['BG_diff'] = df['BG'] - df['BG_ref']

    var_x = 'T_diff'
    var_y = 'BG_diff'
    
    # Prepare data
    X = np.array(df[var_x])
    Y = np.array(df[var_y])

    # Compute slope (no-intercept linear regression)
    beta = np.sum(X * Y) / np.sum(X * X)

    # Predictions
    x_plot = np.linspace(X.min() - 10, X.max() + 10, 100).reshape(-1, 1)
    y_pred = beta * x_plot

    # Residuals and standard error
    residuals = Y - beta * X
    residual_sum_of_squares = np.sum(residuals**2)
    degrees_of_freedom = len(X) - 1
    variance = residual_sum_of_squares / degrees_of_freedom
    std_error_beta = np.sqrt(variance) / np.sqrt(np.sum(X * X))

    # Prediction standard error
    number_of_averaged_samples = 200
    std_error_pred = np.sqrt(variance / (2.*number_of_averaged_samples) + x_plot**2 * variance / np.sum(X * X))

    # Confidence intervals
    t_value = t.ppf(1 - 0.025, degrees_of_freedom)  # 95% confidence
    beta_ci = [
        beta - t_value * std_error_beta,
        beta + t_value * std_error_beta,
    ]
    regression_lower_bound = beta_ci[0] * x_plot
    regression_upper_bound = beta_ci[1] * x_plot

    prediction_lower_bound = 5 - beta * x_plot + t_value * std_error_pred
    prediction_upper_bound = 5 - beta * x_plot - t_value * std_error_pred

    # Print results
    print(f"Residual variance (sigma^2): {variance / 4:.4f}")
    print(f"Estimated beta: {beta:.4f}")
    print(f"95% CI for beta: {beta_ci}")

    # Plot regression
    plt.figure()
    plt.scatter(X, Y, color='blue', label='Data Points')
    plt.plot(x_plot, y_pred, color='red', 
             label=rf"Linear Fit: $BG_{{\text{{diff}}}}$ = {beta:.4f}$T_{{\text{{diff}}}}$")
    plt.fill_between(
        x_plot.flatten(), 
        regression_lower_bound.flatten(), 
        regression_upper_bound.flatten(), 
        color='gray', alpha=0.3, label='95% Regression CI'
    )
    plt.grid(True)
    plt.xlabel(r"$T_{\text{diff}}$ [°C]")
    plt.ylabel(r"$BG_{\text{diff}}$ [mmol/L]")
    plt.xlim([-30, 10])
    plt.legend()
    plt.savefig('BGdiffTdiff.png', dpi=600)

    # Plot predictions
    plt.figure()
    plt.plot(
        x_plot + 22.5, 
        5 - beta * x_plot, 
        color='red', 
        label=r"Maximum-likelihood: $BG_{\text{actual}}$ for $BG_{\text{Contour}} = 5$ mmol/L"
    )
    plt.fill_between(
        (x_plot + 22.5).flatten(), 
        prediction_lower_bound.flatten(), 
        prediction_upper_bound.flatten(), 
        color='gray', alpha=0.3, label='95% Prediction CI'
    )
    plt.grid(True)
    plt.xlabel(r"$T$ [°C]")
    plt.ylabel(r"$BG_{\text{actual}}$ [mmol/L]")
    plt.xlim([-30 + 22.5, 10 + 22.5])
    plt.legend()
    plt.savefig('BGT_'+str(number_of_averaged_samples)+'_samples.png', dpi=600)


# Example Usage
df = pd.read_csv('data.csv') 
fit_BG_curve(df)
plt.show()
