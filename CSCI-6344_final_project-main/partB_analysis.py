"""
partB_analysis.py

Performs Part B: Statistical analysis & modeling on processed F1 telemetry and lap data.

Outputs saved to `partB_results/`.
"""
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve


OUTDIR = 'partB_results'
os.makedirs(OUTDIR, exist_ok=True)


def load_data():
    laps_path = 'processed_laps.csv'
    telem_parquet = 'processed_telemetry.parquet'
    telem_csv = 'processed_telemetry.csv'

    if not os.path.exists(laps_path):
        raise FileNotFoundError(f"{laps_path} not found. Run the data extraction cells first.")

    df_laps = pd.read_csv(laps_path)

    if os.path.exists(telem_parquet):
        all_telemetry = pd.read_parquet(telem_parquet)
    elif os.path.exists(telem_csv):
        all_telemetry = pd.read_csv(telem_csv)
    else:
        raise FileNotFoundError('No telemetry file found (processed_telemetry.parquet/csv).')

    return df_laps, all_telemetry


def aggregate_telemetry(all_telemetry):
    # Ensure expected columns exist
    for col in ['Driver', 'GP', 'Speed', 'Throttle']:
        if col not in all_telemetry.columns:
            raise KeyError(f"Telemetry missing expected column: {col}")

    agg = all_telemetry.groupby(['Driver', 'GP']).agg(
        mean_speed=('Speed', 'mean'),
        std_speed=('Speed', 'std'),
        max_speed=('Speed', 'max'),
        mean_throttle=('Throttle', 'mean'),
        std_throttle=('Throttle', 'std'),
        n_points=('Speed', 'count')
    ).reset_index()
    return agg


def run_ttest_anova(df_laps):
    # T-test Elite vs Midfield on LapTimeDeltaSeconds
    elite = df_laps[df_laps['Tier'] == 'Elite']['LapTimeDeltaSeconds'].dropna()
    midfield = df_laps[df_laps['Tier'] == 'Midfield']['LapTimeDeltaSeconds'].dropna()

    tstat, pval = stats.ttest_ind(elite, midfield, equal_var=False)
    print('T-test Elite vs Midfield: t=%.4f, p=%.4e' % (tstat, pval))

    # One-way ANOVA across Drivers
    model = ols('LapTimeDeltaSeconds ~ C(Driver)', data=df_laps).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print('\nANOVA by Driver:')
    print(anova_table)

    # Post-hoc Tukey (requires enough groups)
    try:
        tukey = pairwise_tukeyhsd(endog=df_laps['LapTimeDeltaSeconds'], groups=df_laps['Driver'], alpha=0.05)
        print('\nTukey HSD summary:')
        print(tukey.summary())
    except Exception as e:
        print('Tukey HSD failed:', e)

    # Save results
    with open(os.path.join(OUTDIR, 'ttest_anova.txt'), 'w') as f:
        f.write(f'T-test Elite vs Midfield: t={tstat:.4f}, p={pval:.6e}\n\n')
        f.write('ANOVA:\n')
        f.write(anova_table.to_string())


def prepare_model_data(df_laps, telem_agg):
    merged = pd.merge(
        df_laps,
        telem_agg,
        how='left',
        on=['Driver', 'GP']
    )
    
    #Only drop rows where the *core* freatures are missing
    required_cols = ['mean_speed', 'SpeedSt']
    existing_required = [c for c in required_cols if c in merged.columns]
    merge = merged.dropna(subset = existing_required)
    
    print("Merged data shape:", merged.shape)
    print("Merged Columns:", merged.columns.tolist())
    return merged

def _get_feature_lsit(merged):
    """Return only telemetry features that exist in the merged dataset."""
    candidate_features = [
        "SpeedSt",
        "mean_speed",
        "std_speed",
        "mean_throttle",
        "std_throttle",
    ]
    return [f for f in candidate_features if f in merged.columns]

def run_regression(merged):
    features = _get_feature_lsit(merged)
    print("Regression features:", features)
    
    X = merged[features]
    y = merged['LapTimeDeltaSeconds']

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print('\nOLS Regression Summary:')
    print(model.summary())

    # Save summary
    with open(os.path.join(OUTDIR, 'ols_summary.txt'), 'w') as f:
        f.write(model.summary().as_text())

    # Residual plot
    plt.figure(figsize=(8, 5))
    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.savefig(os.path.join(OUTDIR, 'residuals_vs_fitted.png'))
    plt.close()


def run_classification(merged):
    # Create binary target: fast lap = LapTimeDeltaSeconds < median
    merged = merged.copy()
    merged['fast'] = (merged['LapTimeDeltaSeconds'] < merged['LapTimeDeltaSeconds'].median()).astype(int)

    features = _get_feature_lsit(merged)
    print("Classification features:", features)
    
    X = merged[features].fillna(0)
    y = merged['fast']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_s, y_train)
    y_pred = logreg.predict(X_test_s)
    y_prob = logreg.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    print('\nLogistic Regression: acc=%.3f, AUC=%.3f' % (acc, auc))
    print('Confusion matrix:\n', cm)

    # Decision Tree
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)
    print('\nDecision Tree Accuracy: %.3f' % acc_dt)

    # Save metrics
    with open(os.path.join(OUTDIR, 'classification_metrics.txt'), 'w') as f:
        f.write(f'Logistic: acc={acc:.4f}, auc={auc:.4f}\n')
        f.write('Confusion matrix:\n')
        f.write(str(cm) + '\n')
        f.write(f'DecisionTree acc={acc_dt:.4f}\n')

    # ROC plot for logistic
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'LogReg AUC={auc:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTDIR, 'roc_logreg.png'))
    plt.close()


def correlation_consistency(merged, all_telemetry):
    # Correlation matrix for numeric features
    base_cols = ['LapTimeDeltaSeconds'] + _get_feature_lsit(merged)
    corr = merged[base_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, 'correlation_matrix.png'))
    plt.close()

    # Consistency: per-driver std and coefficient of variation
    driver_stats = merged.groupby('Driver')['LapTimeDeltaSeconds'].agg(['mean', 'std', 'count']).reset_index()
    driver_stats['cv'] = driver_stats['std'] / driver_stats['mean']
    driver_stats = driver_stats.sort_values('std')
    driver_stats.to_csv(os.path.join(OUTDIR, 'driver_consistency.csv'), index=False)
    print('\nDriver consistency (saved to driver_consistency.csv)')


def main():
    print('Loading data...')
    df_laps, all_telemetry = load_data()
    print('Aggregating telemetry features...')
    telem_agg = aggregate_telemetry(all_telemetry)
    print('Preparing merged dataset...')
    merged = prepare_model_data(df_laps, telem_agg)

    print('\nRunning t-test and ANOVA...')
    run_ttest_anova(df_laps)

    print('\nRunning regression analysis...')
    run_regression(merged)

    print('\nRunning classification models...')
    run_classification(merged)

    print('\nRunning correlation and consistency analysis...')
    correlation_consistency(merged, all_telemetry)

    print('\nPart B analysis complete. Results saved in', OUTDIR)


if __name__ == '__main__':
    main()
