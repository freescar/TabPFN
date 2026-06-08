#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WET EH Process Analysis: Fit delta_MET vs loop_count relationship

Purpose:
  1. Load CSV data
  2. Compute delta_MET = BW092WETEH_MET - BW092EH_MET
  3. Fit multiple models (linear, polynomial, non-linear, ML-based)
  4. Output Top3 R² results + 5 non-linear formulas + 1 fitting matrix
  5. Generate scatter plots and time-series curves for Post_MET prediction

Usage:
  python fit_loop_delta_met_analysis.py --data-path <csv_path> --output-dir <output_dir>

Example:
  python fit_loop_delta_met_analysis.py \
    --data-path ./datasets/A2_DBJOA_BW09_PLUS_20260101_20260601_merge_curr_pre_r2r_post_36tool.csv \
    --output-dir ./results/fit_loop_met
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import curve_fit
from scipy.stats import linregress
import logging

warnings.filterwarnings('ignore')

# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== PLOTTING STYLE ====================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== CUSTOM NON-LINEAR MODELS ====================

def linear_model(x, a, b):
    """Linear: y = a*x + b"""
    return a * x + b

def quadratic_model(x, a, b, c):
    """Quadratic: y = a*x² + b*x + c"""
    return a * x**2 + b * x + c

def cubic_model(x, a, b, c, d):
    """Cubic: y = a*x³ + b*x² + c*x + d"""
    return a * x**3 + b * x**2 + c * x + d

def exponential_model(x, a, b):
    """Exponential: y = a * exp(b*x)"""
    return a * np.exp(b * x)

def power_model(x, a, b):
    """Power: y = a * x^b"""
    return a * np.power(np.abs(x) + 0.1, b)

def logarithmic_model(x, a, b):
    """Logarithmic: y = a * log(x) + b"""
    return a * np.log(np.abs(x) + 1) + b

def sigmoid_model(x, a, b, c):
    """Sigmoid: y = a / (1 + exp(-b*(x-c)))"""
    return a / (1 + np.exp(-b * (x - c)))

def gaussian_model(x, a, mu, sigma):
    """Gaussian: y = a * exp(-(x-mu)²/(2*sigma²))"""
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2))

def reciprocal_model(x, a, b):
    """Reciprocal: y = a / (x + b)"""
    return a / (np.abs(x) + np.abs(b) + 0.01)

# ==================== MODEL FITTING FUNCTIONS ====================

def fit_sklearn_model(X, y, model_type='linear'):
    """Fit sklearn models"""
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0),
        'lasso': Lasso(alpha=0.1),
        'rf': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        'gbm': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3, learning_rate=0.1)
    }
    
    model = models.get(model_type, LinearRegression())
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    
    return model, y_pred, r2, rmse, mae

def fit_scipy_curve(x_data, y_data, model_func, initial_guess, bounds=None, model_name=''):
    """Fit scipy curve_fit models"""
    try:
        if bounds:
            popt, _ = curve_fit(model_func, x_data, y_data, p0=initial_guess, bounds=bounds, maxfev=10000)
        else:
            popt, _ = curve_fit(model_func, x_data, y_data, p0=initial_guess, maxfev=10000)
        
        y_pred = model_func(x_data, *popt)
        r2 = r2_score(y_data, y_pred)
        rmse = np.sqrt(mean_squared_error(y_data, y_pred))
        mae = mean_absolute_error(y_data, y_pred)
        
        return popt, y_pred, r2, rmse, mae, True
    except Exception as e:
        logger.warning(f"Failed to fit {model_name}: {str(e)}")
        return None, None, -999, None, None, False

def get_formula_string(model_type, params):
    """Generate formula strings for different models"""
    formulas = {
        'linear': f"y = {params[0]:.6f}*x + {params[1]:.6f}",
        'quadratic': f"y = {params[0]:.6f}*x² + {params[1]:.6f}*x + {params[2]:.6f}",
        'cubic': f"y = {params[0]:.6f}*x³ + {params[1]:.6f}*x² + {params[2]:.6f}*x + {params[3]:.6f}",
        'exponential': f"y = {params[0]:.6f} * exp({params[1]:.6f}*x)",
        'power': f"y = {params[0]:.6f} * x^{params[1]:.6f}",
        'logarithmic': f"y = {params[0]:.6f} * ln(x) + {params[1]:.6f}",
        'sigmoid': f"y = {params[0]:.6f} / (1 + exp(-{params[1]:.6f}*(x-{params[2]:.6f})))",
        'gaussian': f"y = {params[0]:.6f} * exp(-((x-{params[1]:.6f})²/(2*{params[2]:.6f}²)))",
        'reciprocal': f"y = {params[0]:.6f} / (x + {params[1]:.6f})"
    }
    return formulas.get(model_type, "Unknown formula")

# ==================== MAIN ANALYSIS FUNCTION ====================

def analyze_loop_delta_met(data_path, output_dir):
    """Main analysis function"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # ========== Load Data ==========
    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"[INFO] Loaded data: {len(df)} rows, {len(df.columns)} columns")
    
    # ========== Data Validation ==========
    required_cols = ['loop_count', 'BW092EH_MET', 'BW092WETEH_MET']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return
    
    # ========== Compute delta_MET ==========
    logger.info("Computing delta_MET = BW092WETEH_MET - BW092EH_MET")
    df['delta_MET'] = df['BW092WETEH_MET'] - df['BW092EH_MET']
    
    # Remove NaN values
    df_clean = df[['loop_count', 'BW092EH_MET', 'BW092WETEH_MET', 'delta_MET']].dropna()
    logger.info(f"[INFO] Valid data points after cleaning: {len(df_clean)}")
    
    # ========== Data Statistics ==========
    x_data = df_clean['loop_count'].values.astype(float)
    y_data = df_clean['delta_MET'].values.astype(float)
    
    logger.info(f"       loop_count range: [{x_data.min():.2f}, {x_data.max():.2f}]")
    logger.info(f"       delta_MET range:  [{y_data.min():.6f}, {y_data.max():.6f}]")
    
    # ========== Fit Multiple Models ==========
    results = {}
    
    # 1. Linear Regression
    logger.info("Fitting Linear model...")
    model_lr, y_pred_lr, r2_lr, rmse_lr, mae_lr = fit_sklearn_model(
        x_data.reshape(-1, 1), y_data, 'linear'
    )
    results['linear'] = {
        'r2': r2_lr,
        'rmse': rmse_lr,
        'mae': mae_lr,
        'y_pred': y_pred_lr,
        'params': [model_lr.coef_[0], model_lr.intercept_],
        'formula': f"y = {model_lr.coef_[0]:.6f}*x + {model_lr.intercept_:.6f}",
        'model': model_lr
    }
    logger.info(f"[LINEAR] R² = {r2_lr:.6f}")
    
    # 2. Polynomial Regression (2nd, 3rd order)
    for order in [2, 3]:
        logger.info(f"Fitting Polynomial (order={order}) model...")
        poly_features = PolynomialFeatures(degree=order)
        X_poly = poly_features.fit_transform(x_data.reshape(-1, 1))
        model_poly, y_pred_poly, r2_poly, rmse_poly, mae_poly = fit_sklearn_model(
            X_poly, y_data, 'linear'
        )
        
        # Get polynomial coefficients
        coeffs = model_poly.coef_
        poly_name = f"poly_{order}"
        
        results[poly_name] = {
            'r2': r2_poly,
            'rmse': rmse_poly,
            'mae': mae_poly,
            'y_pred': y_pred_poly,
            'params': coeffs.tolist(),
            'formula': f"Polynomial order {order}",
            'model': model_poly,
            'poly_features': poly_features
        }
        logger.info(f"[POLY_{order}] R² = {r2_poly:.6f}")
    
    # 3. Non-linear Models (scipy curve_fit)
    non_linear_models = [
        ('quadratic', quadratic_model, [1, 1, 5], None),
        ('exponential', exponential_model, [5, 0.1], ([0.1, -1], [20, 1])),
        ('power', power_model, [5, 1], ([0.1, 0.1], [20, 3])),
        ('logarithmic', logarithmic_model, [1, 5], None),
        ('sigmoid', sigmoid_model, [6, 1, 4], ([0.1, 0.1, 1], [20, 5, 7])),
    ]
    
    for model_name, model_func, initial_guess, bounds in non_linear_models:
        logger.info(f"Fitting {model_name} model...")
        popt, y_pred, r2, rmse, mae, success = fit_scipy_curve(
            x_data, y_data, model_func, initial_guess, bounds, model_name
        )
        
        if success:
            results[model_name] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'y_pred': y_pred,
                'params': popt.tolist(),
                'formula': get_formula_string(model_name, popt),
                'model': model_func
            }
            logger.info(f"[{model_name.upper()}] R² = {r2:.6f}")
        else:
            logger.warning(f"Failed to fit {model_name}")
    
    # 4. Machine Learning Models (RF, GBM)
    for ml_model in ['rf', 'gbm']:
        logger.info(f"Fitting {ml_model.upper()} model...")
        model, y_pred, r2, rmse, mae = fit_sklearn_model(
            x_data.reshape(-1, 1), y_data, ml_model
        )
        results[ml_model] = {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'y_pred': y_pred,
            'params': [],
            'formula': f"{ml_model.upper()} Model",
            'model': model
        }
        logger.info(f"[{ml_model.upper()}] R² = {r2:.6f}")
    
    # ========== Gaussian Model ==========
    logger.info("Fitting Gaussian model...")
    try:
        popt_gauss, y_pred_gauss, r2_gauss, rmse_gauss, mae_gauss, success = fit_scipy_curve(
            x_data, y_data, gaussian_model, [6, 4, 1], None, 'gaussian'
        )
        if success:
            results['gaussian'] = {
                'r2': r2_gauss,
                'rmse': rmse_gauss,
                'mae': mae_gauss,
                'y_pred': y_pred_gauss,
                'params': popt_gauss.tolist(),
                'formula': get_formula_string('gaussian', popt_gauss),
                'model': gaussian_model
            }
            logger.info(f"[GAUSSIAN] R² = {r2_gauss:.6f}")
    except:
        logger.warning("Failed to fit Gaussian model")
    
    # ========== Sort and Get Top Results ==========
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    top3_results = sorted_results[:3]
    other_results = sorted_results[3:8]  # Next 5 results
    
    logger.info("\n" + "="*80)
    logger.info("TOP 3 BEST FITS:")
    logger.info("="*80)
    for idx, (model_name, model_data) in enumerate(top3_results, 1):
        logger.info(f"{idx}. {model_name.upper():<15} | R² = {model_data['r2']:.6f} | RMSE = {model_data['rmse']:.6f} | MAE = {model_data['mae']:.6f}")
        logger.info(f"   Formula: {model_data['formula']}")
    
    logger.info("\n" + "="*80)
    logger.info("OTHER NON-LINEAR MODELS (5 additional):")
    logger.info("="*80)
    for idx, (model_name, model_data) in enumerate(other_results[:5], 1):
        logger.info(f"{idx}. {model_name.upper():<15} | R² = {model_data['r2']:.6f} | RMSE = {model_data['rmse']:.6f} | MAE = {model_data['mae']:.6f}")
        logger.info(f"   Formula: {model_data['formula']}")
    
    # ========== Save Results to JSON ==========
    output_json = os.path.join(output_dir, 'fit_loop_delta_met_results.json')
    
    output_data = {
        'summary': {
            'total_samples': len(df_clean),
            'loop_count_range': [float(x_data.min()), float(x_data.max())],
            'delta_met_range': [float(y_data.min()), float(y_data.max())]
        },
        'top_3_models': {
            model_name: {
                'r2': float(model_data['r2']),
                'rmse': float(model_data['rmse']),
                'mae': float(model_data['mae']),
                'formula': model_data['formula'],
                'params': [float(p) if isinstance(p, (np.floating, float)) else p for p in model_data['params']]
            }
            for model_name, model_data in top3_results
        },
        'other_5_models': {
            model_name: {
                'r2': float(model_data['r2']),
                'rmse': float(model_data['rmse']),
                'mae': float(model_data['mae']),
                'formula': model_data['formula'],
                'params': [float(p) if isinstance(p, (np.floating, float)) else p for p in model_data['params']]
            }
            for model_name, model_data in other_results[:5]
        },
        'all_models_ranked': {
            model_name: {
                'r2': float(model_data['r2']),
                'rmse': float(model_data['rmse']),
                'mae': float(model_data['mae'])
            }
            for model_name, model_data in sorted_results
        }
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    logger.info(f"\n[SAVED] Results JSON: {output_json}")
    
    # ========== Generate Scatter Plots for Top 3 ==========
    logger.info("\nGenerating scatter plots for Top 3 models...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Top 3 Model Fits: delta_MET vs loop_count', fontsize=16, fontweight='bold')
    
    for idx, (ax, (model_name, model_data)) in enumerate(zip(axes, top3_results)):
        # Sort for better visualization
        sort_idx = np.argsort(x_data)
        x_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]
        y_pred_sorted = model_data['y_pred'][sort_idx]
        
        # Scatter plot
        ax.scatter(x_sorted, y_sorted, s=50, alpha=0.5, label='Actual', color='steelblue')
        
        # Fitted curve
        ax.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2.5, label='Fitted')
        
        # Formatting
        ax.set_xlabel('loop_count', fontsize=11, fontweight='bold')
        ax.set_ylabel('delta_MET (Å)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model_name.upper()}\nR² = {model_data["r2"]:.6f}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Add formula text
        formula_text = model_data['formula']
        ax.text(0.05, 0.95, formula_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    scatter_plot_path = os.path.join(output_dir, 'top3_scatter_plots.png')
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"[SAVED] Scatter plots: {scatter_plot_path}")
    plt.close()
    
    # ========== Generate Time-Series Plot for Post_MET Prediction ==========
    logger.info("\nGenerating time-series plot for Post_MET prediction...")
    
    # Use the best model for prediction
    best_model_name, best_model_data = top3_results[0]
    
    # Calculate Post_MET prediction: post_MET = pre_MET + delta_MET
    pre_met = df_clean['BW092EH_MET'].values
    actual_post_met = df_clean['BW092WETEH_MET'].values
    predicted_delta_met = best_model_data['y_pred']
    predicted_post_met = pre_met + predicted_delta_met
    
    # Sort by index for time-series visualization
    sample_indices = np.arange(len(predicted_post_met))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Time-series of delta_MET
    ax1.plot(sample_indices, y_data, 'b-', alpha=0.7, linewidth=1.5, label='Actual delta_MET')
    ax1.plot(sample_indices, predicted_delta_met, 'r--', alpha=0.7, linewidth=1.5, label='Predicted delta_MET')
    ax1.fill_between(sample_indices, y_data, predicted_delta_met, alpha=0.1)
    ax1.set_ylabel('delta_MET (Å)', fontsize=11, fontweight='bold')
    ax1.set_title(f'delta_MET Prediction (Model: {best_model_name})', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time-series of Post_MET
    ax2.plot(sample_indices, actual_post_met, 'g-', alpha=0.7, linewidth=1.5, label='Actual Post_MET')
    ax2.plot(sample_indices, predicted_post_met, 'orange', linestyle='--', alpha=0.7, linewidth=1.5, label='Predicted Post_MET')
    ax2.fill_between(sample_indices, actual_post_met, predicted_post_met, alpha=0.1)
    ax2.set_xlabel('Sample Index', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Post_MET (Å)', fontsize=11, fontweight='bold')
    ax2.set_title('Post_MET Prediction (post_MET = pre_MET + delta_MET)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timeseries_plot_path = os.path.join(output_dir, 'post_met_timeseries_prediction.png')
    plt.savefig(timeseries_plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"[SAVED] Time-series plot: {timeseries_plot_path}")
    plt.close()
    
    # ========== Calculate Post_MET Prediction Metrics ==========
    r2_post_met = r2_score(actual_post_met, predicted_post_met)
    rmse_post_met = np.sqrt(mean_squared_error(actual_post_met, predicted_post_met))
    mae_post_met = mean_absolute_error(actual_post_met, predicted_post_met)
    
    logger.info("\n" + "="*80)
    logger.info("POST_MET PREDICTION METRICS (indirect via delta_MET):")
    logger.info("="*80)
    logger.info(f"R² Score:  {r2_post_met:.6f}")
    logger.info(f"RMSE:      {rmse_post_met:.6f}")
    logger.info(f"MAE:       {mae_post_met:.6f}")
    
    # ========== Save Comprehensive Report ==========
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("WET EH PROCESS ANALYSIS: delta_MET vs loop_count\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATA SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(f"Total samples: {len(df_clean)}\n")
        f.write(f"loop_count range: [{x_data.min():.2f}, {x_data.max():.2f}]\n")
        f.write(f"delta_MET range: [{y_data.min():.6f}, {y_data.max():.6f}]\n\n")
        
        f.write("TOP 3 BEST FITS\n")
        f.write("-"*80 + "\n")
        for idx, (model_name, model_data) in enumerate(top3_results, 1):
            f.write(f"{idx}. {model_name.upper()}\n")
            f.write(f"   R² = {model_data['r2']:.6f}\n")
            f.write(f"   RMSE = {model_data['rmse']:.6f}\n")
            f.write(f"   MAE = {model_data['mae']:.6f}\n")
            f.write(f"   Formula: {model_data['formula']}\n\n")
        
        f.write("OTHER 5 NON-LINEAR MODELS\n")
        f.write("-"*80 + "\n")
        for idx, (model_name, model_data) in enumerate(other_results[:5], 1):
            f.write(f"{idx}. {model_name.upper()}\n")
            f.write(f"   R² = {model_data['r2']:.6f}\n")
            f.write(f"   RMSE = {model_data['rmse']:.6f}\n")
            f.write(f"   MAE = {model_data['mae']:.6f}\n")
            f.write(f"   Formula: {model_data['formula']}\n\n")
        
        f.write("POST_MET PREDICTION (Indirect via delta_MET)\n")
        f.write("-"*80 + "\n")
        f.write(f"Using best model: {best_model_name.upper()}\n")
        f.write(f"Strategy: post_MET = pre_MET + delta_MET\n")
        f.write(f"R² Score: {r2_post_met:.6f}\n")
        f.write(f"RMSE: {rmse_post_met:.6f}\n")
        f.write(f"MAE: {mae_post_met:.6f}\n")
    
    logger.info(f"[SAVED] Analysis report: {report_path}")
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE!")
    logger.info("="*80)
    
    return output_data

# ==================== MAIN ENTRY POINT ====================

def main():
    parser = argparse.ArgumentParser(
        description='Analyze WET EH process: fit delta_MET vs loop_count relationship'
    )
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--output-dir', type=str, default='./results/fit_loop_met',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        return
    
    analyze_loop_delta_met(args.data_path, args.output_dir)

if __name__ == '__main__':
    main()
