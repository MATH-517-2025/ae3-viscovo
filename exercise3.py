#| echo: false
#| message: false
#| warning: false

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

# Set seed for reproducibility
np.random.seed(42)

# --- GLOBAL PARAMETERS ---
SIGMA2_BASE = 1.0
R_REPETITIONS = 50 # Number of repetitions for stable estimates

# --- 1. CORE FUNCTIONS ---

# Regression function (using user's specified version)
reg_function = lambda x: np.sin(1 / (x/3 + 0.1)) 

def beta_sample_generator(sample_size, alfa, beta, error_variance=SIGMA2_BASE, error_mean=0):
    """Generates i.i.d. samples (X, Y) based on Beta distribution and the m(x) function."""
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
    
    # 1. Define block boundaries (N+1 points including 0 and n)
    cut_indices = np.round(np.linspace(0, n, N + 1))

    RSS_N = 0 
    theta2_num_sum = 0
    
    # 2. Loop over N blocks
    for j in range(N):
        start_index = int(cut_indices[j])
        end_index = int(cut_indices[j+1])
        
        if start_index >= end_index: continue
            
        X_j = x_sorted[start_index:end_index]
        Y_j = y_sorted[start_index:end_index]
        
        # 3. Create Design Matrix X_j (Polynomial Degree 4)
        col_ones = np.ones_like(X_j) 
        X_j_matrix = np.column_stack([col_ones, X_j, X_j ** 2, X_j ** 3, X_j ** 4])
        
        # 4. Calculate OLS coefficients (beta_j)
        try:
            beta_j, residuals_sum_of_squares, rank, singular_values = lstsq(X_j_matrix, Y_j, rcond=None)
        except np.linalg.LinAlgError:
            return np.nan, np.nan 

        # Extract coefficients for the second derivative
        beta2, beta3, beta4 = beta_j[2], beta_j[3], beta_j[4]
        
        # 5. Aggregate RSS (for sigma^2)
        if residuals_sum_of_squares.size > 0:
             RSS_N += residuals_sum_of_squares[0]
        
        # 6. Aggregate Squared Second Derivative (for theta_2^2)
        m_double_prime_j = 2 * beta2 + 6 * beta3 * X_j + 12 * beta4 * (X_j ** 2)
        theta2_num_sum += np.sum(m_double_prime_j ** 2)

    # 7. Final h_AMISE Calculation
    df_denominator = n - 5 * N
    if df_denominator <= 0 or theta2_num_sum == 0:
        return np.nan, np.nan 

    sigma2_hat = RSS_N / df_denominator
    theta2_hat = theta2_num_sum / n
    
    if theta2_hat <= 1e-9: 
        return np.nan, np.nan
        
    numerator = 35 * sigma2_hat
    denominator = 1 * theta2_hat # |supp(X)| = 1
    
    h_AMISE_hat = n**(-1/5) * (numerator / denominator)**(1/5)
    
    return h_AMISE_hat, RSS_N

def calculate_n_opt(x, y):
    """Calculates N_opt by minimizing Mallows' Cp criterion and returns all relevant metrics."""
    n = len(x)
    
    # N_max = max{min(⌊n/20⌋, 5), 1}
    N_max = max(min(n // 20, 5), 1)

    N_values = np.arange(1, N_max + 1)
    RSS_values = []
    h_AMISE_values = []
    
    # 1. Calculate RSS(N) and h_AMISE(N) for N = 1 to N_max
    for N in N_values:
        h, rss = opt_bandwidth(x, y, N)
        h_AMISE_values.append(h)
        RSS_values.append(rss)

    RSS_values = np.array(RSS_values)
    h_AMISE_values = np.array(h_AMISE_values)

    # 2. Calculate sigma^2_ref using N_max
    RSS_N_max = RSS_values[-1]
    df_N_max = n - 5 * N_max
    
    if df_N_max <= 0: return np.nan, np.nan, np.nan, np.nan
        
    sigma2_ref = RSS_N_max / df_N_max
    
    # 3. Calculate Cp(N)
    Cp_values = (RSS_values / sigma2_ref) - (n - 10 * N_values)
        
    # 4. Find N_opt
    min_cp_index = np.argmin(Cp_values)
    N_opt = N_values[min_cp_index]
    
    return N_opt, h_AMISE_values, Cp_values, N_values

# --- RUN ALL EXPERIMENTS ---

# Experiment 1: Impact of Number of Blocks (N)
N_EXP_N_SAMPLE = 500
h_amise_N_matrix = []
Cp_N_matrix = []

for r in range(R_REPETITIONS):
    X_base, Y_base = beta_sample_generator(N_EXP_N_SAMPLE, 1, 1)
    N_opt_r, h_amise_r, Cp_r, N_vals_r = calculate_n_opt(X_base, Y_base)
    
    if len(h_amise_r) == len(N_vals_r): 
        h_amise_N_matrix.append(h_amise_r)
        Cp_N_matrix.append(Cp_r)

# Calculate means
h_amise_N_mean = np.mean(h_amise_N_matrix, axis=0)
Cp_N_mean = np.mean(Cp_N_matrix, axis=0)
N_opt_mean = N_vals_r[np.argmin(Cp_N_mean)]

# Experiment 2: Impact of Sample Size (n)
N_GRID = [100, 250, 500, 1000, 2000]
h_amise_n_means = []

for n_val in N_GRID:
    h_amise_rep = []
    for r in range(R_REPETITIONS):
        X, Y = beta_sample_generator(n_val, 1, 1)
        N_opt_r, h_amise_r, Cp_r, N_vals_r = calculate_n_opt(X, Y)
        if not np.isnan(N_opt_r):
             h_amise_at_n_opt = h_amise_r[np.where(N_vals_r == N_opt_r)[0][0]]
             h_amise_rep.append(h_amise_at_n_opt)
    h_amise_n_means.append(np.mean(h_amise_rep))

# Convert to log scale for linearity check
log_n = np.log(N_GRID)
log_h = np.log(h_amise_n_means)
coefficients = np.polyfit(log_n, log_h, 1)
slope = coefficients[0]

# Experiment 3: Impact of Covariate Density Shape
BETA_PARAMS = {
    'Uniform (1, 1)': (1, 1),
    'Symmetric Center (5, 5)': (5, 5),
    'Skewed Right (5, 2)': (5, 2),
    'Skewed Left (2, 5)': (2, 5),
    'U-Shaped (0.5, 0.5)': (0.5, 0.5)
}
DENSITY_EXP_N_SAMPLE = 1000
DENSITY_EXP_N_BLOCKS = 5 

h_amise_density_means = {}
for label, (alpha, beta) in BETA_PARAMS.items():
    h_amise_rep = []
    for r in range(R_REPETITIONS):
        X, Y = beta_sample_generator(DENSITY_EXP_N_SAMPLE, alpha, beta)
        h_amise_r, _ = opt_bandwidth(X, Y, DENSITY_EXP_N_BLOCKS)
        if not np.isnan(h_amise_r):
            h_amise_rep.append(h_amise_r)
    h_amise_density_means[label] = np.mean(h_amise_rep)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Impact of Number of Blocks (N)
plt.sca(axes[0, 0])
plt.plot(N_vals_r, h_amise_N_mean, marker='o', linestyle='-', color='blue', linewidth=2)
plt.axvline(N_opt_mean, color='red', linestyle='--', label=f'$N_{{opt}} = {N_opt_mean}$')
plt.title('Impact of Number of Blocks on Optimal Bandwidth', fontsize=12, fontweight='bold')
plt.xlabel('Number of Blocks ($N$)')
plt.ylabel(r'$\hat{h}_{AMISE}$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'Minimum h_AMISE at N={N_opt_mean}: {h_amise_N_mean[N_opt_mean-1]:.4f}', 
         transform=axes[0, 0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

# Plot 2: Mallows' Cp Criterion
plt.sca(axes[0, 1])
plt.plot(N_vals_r, Cp_N_mean, marker='s', linestyle='-', color='green', linewidth=2)
plt.axvline(N_opt_mean, color='red', linestyle='--', label=f'$N_{{opt}} = {N_opt_mean}$')
plt.title("Mallows' Cp Criterion vs Number of Blocks", fontsize=12, fontweight='bold')
plt.xlabel('Number of Blocks ($N$)')
plt.ylabel(r'$C_p(N)$')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Impact of Sample Size
plt.sca(axes[1, 0])
plt.scatter(log_n, log_h, color='purple', s=80, label='Observed values')
plt.plot(log_n, coefficients[1] + coefficients[0] * log_n, color='orange', 
         linestyle='--', linewidth=2, label=f'Fit line (slope: {slope:.3f})')
plt.title('Impact of Sample Size on Optimal Bandwidth', fontsize=12, fontweight='bold')
plt.xlabel(r'$\log(n)$')
plt.ylabel(r'$\log(\hat{h}_{AMISE})$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.text(0.05, 0.95, f'Theoretical slope: -0.200\nEstimated slope: {slope:.3f}', 
         transform=axes[1, 0].transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

# Plot 4: Impact of Covariate Density Shape
plt.sca(axes[1, 1])
labels = list(h_amise_density_means.keys())
values = list(h_amise_density_means.values())
colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral', 'plum']
bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.title('Impact of Covariate Distribution Shape on Optimal Bandwidth', fontsize=12, fontweight='bold')
plt.ylabel(r'$\hat{h}_{AMISE}$')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Summary table of results
print("\n" + "="*70)
print("SUMMARY OF SIMULATION RESULTS")
print("="*70)
print(f"\n1. OPTIMAL BLOCK SIZE (N):")
print(f"   Optimal number of blocks: N_opt = {N_opt_mean}")
print(f"   Corresponding h_AMISE: {h_amise_N_mean[N_opt_mean-1]:.4f}")

print(f"\n2. SAMPLE SIZE IMPACT:")
print(f"   Theoretical decay rate: n^(-1/5) = -0.200")
print(f"   Estimated decay rate: {slope:.3f}")

print(f"\n3. DENSITY SHAPE IMPACT:")
for label, h_val in h_amise_density_means.items():
    print(f"   {label:25}: h_AMISE = {h_val:.4f}")