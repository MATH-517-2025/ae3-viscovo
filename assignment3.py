import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
from numpy.linalg import lstsq

# Set seed for reproducibility
np.random.seed(42)

# --- GLOBAL PARAMETERS ---
SIGMA2_BASE = 1.0
R_REPETITIONS = 50  

# --- 1. CORE FUNCTIONS ---

reg_function = lambda x: np.sin(1 / (x/3 + 0.1)) 

def beta_sample_generator(sample_size, alfa, beta, error_variance=SIGMA2_BASE, error_mean=0):
    x = np.random.beta(alfa, beta, sample_size)
    error_std = np.sqrt(error_variance)
    epsilon = np.random.normal(error_mean, error_std, sample_size)
    y = reg_function(x) + epsilon
    return x, y

def opt_bandwidth(x, y, N): 
    """
    Estimates h_AMISE and RSS(N) using the plug-in method with N blocks and 
    OLS polynomial regression of degree 4.
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    n = len(x)
    
    cut_indices = np.round(np.linspace(0, n, N + 1))

    RSS_N = 0 
    theta2_num_sum = 0
    
    for j in range(N):
        start_index = int(cut_indices[j])
        end_index = int(cut_indices[j+1])
        
        if (end_index - start_index) < 5: continue 
            
        X_j = x_sorted[start_index:end_index]
        Y_j = y_sorted[start_index:end_index]
        
        col_ones = np.ones_like(X_j) 
        X_j_matrix = np.column_stack([col_ones, X_j, X_j ** 2, X_j ** 3, X_j ** 4])
        
        beta_j, residuals_sum_of_squares, rank, singular_values = lstsq(X_j_matrix, Y_j, rcond=None)

        beta2, beta3, beta4 = beta_j[2], beta_j[3], beta_j[4]
        
        predictions_j = X_j_matrix @ beta_j
        RSS_j = np.sum((Y_j - predictions_j) ** 2)
        RSS_N += RSS_j

        m_double_prime_j = 2 * beta2 + 6 * beta3 * X_j + 12 * beta4 * (X_j ** 2)
        theta2_num_sum += np.sum(m_double_prime_j ** 2)

    df_denominator = n - 5 * N
    sigma2_hat = RSS_N / df_denominator
    theta2_hat = theta2_num_sum / n
        
    numerator = 35 * sigma2_hat
    denominator = 1 * theta2_hat 
    
    if denominator <= 0 or numerator <= 0:
        return np.nan, RSS_N
        
    h_AMISE_hat = n**(-1/5) * (numerator / denominator)**(1/5)
    
    return h_AMISE_hat, RSS_N

def calculate_n_opt(x, y):
    n = len(x)
    
    N_max = max(min(n // 20, 5), 1)

    N_values = np.arange(1, N_max + 1)
    RSS_values = []
    h_AMISE_values = []
    
    for N in N_values:
        h, rss = opt_bandwidth(x, y, N)
        h_AMISE_values.append(h)
        RSS_values.append(rss)

    RSS_values = np.array(RSS_values)
    h_AMISE_values = np.array(h_AMISE_values)
    
    if np.any(np.isnan(h_AMISE_values)):
        return np.nan, h_AMISE_values, np.full_like(RSS_values, np.nan), N_values

    RSS_N_max = RSS_values[-1]
    df_N_max = n - 5 * N_max
            
    sigma2_ref = RSS_N_max / df_N_max
    
    if sigma2_ref <= 0:
        return np.nan, h_AMISE_values, np.full_like(RSS_values, np.nan), N_values
    
    Cp_values = (RSS_values / sigma2_ref) - (n - 10 * N_values)
        
    min_cp_index = np.argmin(Cp_values)
    N_opt = N_values[min_cp_index]
    
    return N_opt, h_AMISE_values, Cp_values, N_values

# --- 2. RUN SIMULATIONS ---

# Experiment 1: Impact of Number of Blocks (N) - INCREASED SAMPLE SIZE TO 2000
N_EXP_N_SAMPLE = 2000
h_amise_N_matrix = []
Cp_N_matrix = []

for r in range(R_REPETITIONS):
    X_base, Y_base = beta_sample_generator(N_EXP_N_SAMPLE, 1, 1)
    N_opt_r, h_amise_r, Cp_r, N_vals_r = calculate_n_opt(X_base, Y_base)
    
    if not np.any(np.isnan(h_amise_r)) and not np.any(np.isinf(h_amise_r)):
        h_amise_N_matrix.append(h_amise_r)
        Cp_N_matrix.append(Cp_r)

h_amise_N_mean = np.mean(h_amise_N_matrix, axis=0)
Cp_N_mean = np.mean(Cp_N_matrix, axis=0)
N_opt_mean = N_vals_r[np.argmin(Cp_N_mean)]
idx_opt = np.argmin(Cp_N_mean)
h_amise_at_opt = h_amise_N_mean[idx_opt]


# Experiment 2: Impact of Sample Size (n) - ADJUSTED N_GRID
N_GRID = [2000, 3000, 4000, 5000, 7500] 
h_amise_n_means = []

for n_val in N_GRID:
    h_amise_rep = []
    for r in range(R_REPETITIONS):
        X, Y = beta_sample_generator(n_val, 1, 1)
        N_opt_r, h_amise_r, Cp_r, N_vals_r = calculate_n_opt(X, Y)
        
        if np.isfinite(N_opt_r):
            idx = np.where(N_vals_r == N_opt_r)[0][0] 
            h_amise_at_n_opt = h_amise_r[idx]
            
            if np.isfinite(h_amise_at_n_opt):
                 h_amise_rep.append(h_amise_at_n_opt)
                 
    if len(h_amise_rep) > 0:
        h_amise_n_means.append(np.mean(h_amise_rep))
    else:
        h_amise_n_means.append(np.nan)

h_amise_n_means = np.array(h_amise_n_means)

valid_indices = np.isfinite(h_amise_n_means)
N_GRID_VALID = np.array(N_GRID)[valid_indices]
h_amise_VALID = h_amise_n_means[valid_indices]

if len(N_GRID_VALID) >= 2:
    log_n_fit = np.log(N_GRID_VALID) 
    log_h_fit = np.log(h_amise_VALID) 
    coefficients = np.polyfit(log_n_fit, log_h_fit, 1)
    slope = coefficients[0]
else:
    slope = np.nan
    coefficients = [np.nan, np.nan]

log_n_plot = np.log(N_GRID)
log_h_plot = np.log(np.where(h_amise_n_means > 0, h_amise_n_means, np.nan)) 


# Experiment 3: Impact of Covariate Density Shape
BETA_PARAMS = {
    'Uniform (1, 1)': (1, 1),
    'Symmetric Center (5, 5)': (5, 5),
    'Skewed Right (5, 2)': (5, 2),
    'Skewed Left (2, 5)': (2, 5),
    'U-Shaped (0.5, 0.5)': (0.5, 0.5)
}
DENSITY_EXP_N_SAMPLE = 2000 
DENSITY_EXP_N_BLOCKS = 5 

h_amise_density_means = {}
for label, (alpha, beta) in BETA_PARAMS.items():
    h_amise_rep = []
    for r in range(R_REPETITIONS):
        X, Y = beta_sample_generator(DENSITY_EXP_N_SAMPLE, alpha, beta)
        h_amise_r, _ = opt_bandwidth(X, Y, DENSITY_EXP_N_BLOCKS)
        if np.isfinite(h_amise_r):
            h_amise_rep.append(h_amise_r)
    
    if len(h_amise_rep) > 0:
        h_amise_density_means[label] = np.mean(h_amise_rep)
    else:
        h_amise_density_means[label] = np.nan


# --- 3. PLOT FUNCTIONS ---

def plot_h_amise_vs_n(N_vals_r, h_amise_N_mean, N_opt_mean, h_amise_at_opt):
    """Plot 1: Impact of Number of Blocks on h_AMISE"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    ax.plot(N_vals_r, h_amise_N_mean, marker='o', linestyle='-', color='blue', linewidth=2)
    ax.axvline(N_opt_mean, color='red', linestyle='--', label=f'$N_{{opt}} = {N_opt_mean}$')
    ax.scatter(N_opt_mean, h_amise_at_opt, color='red', s=100, zorder=5) 

    ax.set_title(r'Impact of N on Optimal Bandwidth ($\hat{h}_{AMISE}$) [n=2000]', fontsize=14, fontweight='bold')
    ax.set_xlabel(r'Number of Blocks ($N$)', fontsize=12)
    ax.set_ylabel(r'$\hat{h}_{AMISE}$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.4)
    ax.text(0.05, 0.9, rf'$\hat{{h}}_{{AMISE}}$ at $N_{{opt}}$: {h_amise_at_opt:.4f}', 
             transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
    
    plt.savefig('h_amise_vs_N.png')
    plt.close() 

def plot_cp_criterion(N_vals_r, Cp_N_mean, N_opt_mean, idx_opt):
    """Plot 2: Mallows' Cp Criterion"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    ax.plot(N_vals_r, Cp_N_mean, marker='s', linestyle='-', color='green', linewidth=2)
    ax.axvline(N_opt_mean, color='red', linestyle='--', label=f'$N_{{opt}} = {N_opt_mean}$')
    ax.scatter(N_opt_mean, Cp_N_mean[idx_opt], color='red', s=100, zorder=5) 
    
    ax.set_title(r"Mallows' $C_p$ Criterion vs Number of Blocks [n=2000]", fontsize=14, fontweight='bold')
    ax.set_xlabel(r'Number of Blocks ($N$)', fontsize=12)
    ax.set_ylabel(r'$C_p(N)$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.savefig('cp_criterion.png')
    plt.close()

def plot_sample_size_decay(log_n_plot, log_h_plot, slope, log_n_fit, coefficients):
    """Plot 3: Impact of Sample Size (log(h) vs log(n))"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    ax.scatter(log_n_plot, log_h_plot, color='purple', s=80, label='Observed values')

    if np.isfinite(slope):
        ax.plot(log_n_fit, coefficients[1] + coefficients[0] * log_n_fit, color='orange', 
                 linestyle='--', linewidth=2, label=f'Fit line (slope: {slope:.3f})')
    else:
        ax.text(0.5, 0.5, 'Regression Failed (Insufficient valid data)', 
                 transform=ax.transAxes, color='red', ha='center', fontsize=12)

    ax.set_title(r'Decay of $\hat{h}_{AMISE}$ with Sample Size ($n$)', fontsize=14, fontweight='bold')
    ax.set_xlabel(r'$\log(n)$', fontsize=12)
    ax.set_ylabel(r'$\log(\hat{h}_{AMISE})$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.savefig('sample_size_decay.png')
    plt.close()

def plot_density_impact(h_amise_density_means):
    """Plot 4: Impact of Covariate Density Shape"""
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    
    labels = list(h_amise_density_means.keys())
    values = list(h_amise_density_means.values())
    colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'plum']
    bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.9)

    for bar, value in zip(bars, values):
        if np.isfinite(value):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_title(r'Impact of Covariate Density on $\hat{h}_{AMISE}$ [n=2000]', fontsize=14, fontweight='bold')
    ax.set_ylabel(r'$\hat{h}_{AMISE}$', fontsize=12)
    ax.tick_params(axis='x', rotation=45) 
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    
    plt.savefig('density_impact.png')
    plt.close()

# --- 4. GENERATE PLOTS ---

# Plots for Experiment 1: Impact of N and Cp
plot_h_amise_vs_n(N_vals_r, h_amise_N_mean, N_opt_mean, h_amise_at_opt)
plot_cp_criterion(N_vals_r, Cp_N_mean, N_opt_mean, idx_opt)

# Plot for Experiment 2: Impact of Sample Size (n)
plot_sample_size_decay(log_n_plot, log_h_plot, slope, log_n_fit, coefficients)

# Plot for Experiment 3: Impact of Covariate Density Shape
plot_density_impact(h_amise_density_means)