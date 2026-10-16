"""
Train and compare proxy models using all 100 cleaned reservoir cases.

Split:
  Cases 1-70   -> training
  Cases 71-85  -> validation/model selection
  Cases 86-100 -> final test

Models:
  1. PCA + Gaussian Process
  2. PCA + Polynomial degree-2 + Ridge
  3. PCA + Neural Network

Important:
- Each simulation case is ONE model row. Grid-cell rows are summarized into
  case-level features; they are not treated as independent cases.
- Historical cumulative production curves from 02 Field Production.xlsx are
  the targets (1998-2008). This script does NOT create future 2008-2028 targets.
- Existing model_training_results files are NOT deleted. Results go to a new
  folder: proxy_training_grid_features.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
CLEANED_DIR = BASE_DIR / "Cleaned_Data"
PRODUCTION_FILE = BASE_DIR / "Original_Data" / "02 Field Production.xlsx"
# Convenience fallbacks: also work if the workbooks are kept beside this script.
if not PRODUCTION_FILE.exists() and (BASE_DIR / "02 Field Production.xlsx").exists():
    PRODUCTION_FILE = BASE_DIR / "02 Field Production.xlsx"
if not CLEANED_DIR.exists() and list(BASE_DIR.glob("*_Cleaned.xlsx")):
    CLEANED_DIR = BASE_DIR
OUTPUT_DIR = BASE_DIR / "proxy_training_grid_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CASES = list(range(1, 71))
VALIDATION_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))

PRODUCTION_SHEETS = [
    "Gas production cumulative",
    "Oil production cumulative",
    "Water production cumulative",
]
TARGET_LABELS = ["Gas", "Oil", "Water"]
TARGET_DATES = pd.date_range("1998-01-01", "2008-01-01", freq="QS")
MAX_PCA_COMPONENTS = 20

# These columns are categorical region identifiers. They are summarized as
# proportions instead of treating numeric codes as continuous values.
CATEGORICAL_COLUMNS = [
    "Fluid In Place Region (FIPNUM)",
    "PVT Region (PVTNUM)",
]
INDEX_COLUMNS = ["I Index", "J Index", "K Index"]
COORD_COLUMNS = ["X Coordinate", "Y Coordinate", "Z Coordinate"]
SPECIAL_SUM_COLUMNS = ["Pore Volume (PORV)"]


def case_number_from_sheet(sheet_name):
    import re
    match = re.search(r"(\d+)", str(sheet_name))
    return int(match.group(1)) if match else None


def find_case_sheet_files():
    if not CLEANED_DIR.exists():
        raise FileNotFoundError(
            f"Cannot find folder: {CLEANED_DIR}\n"
            "Put the six *_Cleaned.xlsx workbooks in a folder named Cleaned_Data."
        )
    files = sorted(CLEANED_DIR.glob("*_Cleaned.xlsx"))
    if not files:
        raise FileNotFoundError(f"No *_Cleaned.xlsx files found in {CLEANED_DIR}")
    return files


def summarize_case(df, case_no, source_file):
    """Create one physically interpretable feature row from grid-cell rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    row = {"Case": int(case_no), "Source_File": source_file, "Grid_Cell_Count": len(df)}

    # Convert the placeholder used in the source data to missing if it remains.
    df = df.replace(-999.25, np.nan)

    # Grid indices: summarize extent/number of unique cells, not mean as a
    # continuous physical property.
    for col in INDEX_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                row[f"{col}__min"] = float(s.min())
                row[f"{col}__max"] = float(s.max())
                row[f"{col}__nunique"] = int(s.nunique())

    # Coordinates describe model geometry. Summaries are useful, but are not
    # assumed to directly determine production.
    for col in COORD_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                for stat, val in [
                    ("mean", s.mean()), ("std", s.std()),
                    ("min", s.min()), ("max", s.max())
                ]:
                    row[f"{col}__{stat}"] = float(val) if pd.notna(val) else 0.0

    # Region identifiers are categorical: use fractions per region.
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s):
                counts = s.value_counts(normalize=True)
                for region, fraction in counts.items():
                    row[f"{col}__fraction_{region:g}"] = float(fraction)

    # Continuous properties: mean, standard deviation, median, and selected
    # quantiles/range. This retains more information than a single mean.
    excluded = set(INDEX_COLUMNS + COORD_COLUMNS + CATEGORICAL_COLUMNS)
    for col in df.columns:
        if col in excluded:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        s = s.replace([np.inf, -np.inf], np.nan).dropna()
        if s.empty:
            continue

        stats = {
            "mean": s.mean(),
            "std": s.std(),
            "median": s.median(),
            "p10": s.quantile(0.10),
            "p90": s.quantile(0.90),
            "min": s.min(),
            "max": s.max(),
        }
        for stat, val in stats.items():
            if pd.notna(val):
                row[f"{col}__{stat}"] = float(val)

        if col in SPECIAL_SUM_COLUMNS:
            row[f"{col}__sum"] = float(s.sum())

    return row


def load_case_features():
    rows = []
    seen = set()
    files = find_case_sheet_files()
    print("Reading cleaned grid-cell workbooks and summarizing cases...")
    for file in files:
        print(f"  {file.name}")
        xls = pd.ExcelFile(file)
        for sheet in xls.sheet_names:
            case_no = case_number_from_sheet(sheet)
            if case_no is None or not 1 <= case_no <= 100:
                continue
            if case_no in seen:
                raise ValueError(f"Case {case_no} appears in more than one cleaned workbook.")
            df = pd.read_excel(file, sheet_name=sheet)
            if len(df.columns) != 29:
                raise ValueError(
                    f"{file.name} / {sheet}: expected 29 columns, got {len(df.columns)}."
                )
            rows.append(summarize_case(df, case_no, file.name))
            seen.add(case_no)

    expected = set(range(1, 101))
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"Expected cases 1-100. Missing={missing}; unexpected={extra}")

    feature_df = pd.DataFrame(rows).sort_values("Case").reset_index(drop=True)
    metadata_cols = ["Case", "Source_File", "Grid_Cell_Count"]
    feature_cols = [c for c in feature_df.columns if c not in metadata_cols]
    X_df = feature_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # Remove features that are completely missing or constant across cases.
    keep_cols = []
    for col in X_df.columns:
        if X_df[col].notna().sum() == 0:
            continue
        if X_df[col].nunique(dropna=True) <= 1:
            continue
        keep_cols.append(col)
    X_df = X_df[keep_cols]

    if X_df.shape[1] == 0:
        raise ValueError("No usable numeric case-level features were generated.")

    feature_df.to_csv(OUTPUT_DIR / "case_level_feature_summary.csv", index=False)
    print(f"  Cases summarized: {len(feature_df)}")
    print(f"  Usable features before imputation: {X_df.shape[1]}")
    return feature_df["Case"].to_numpy(int), X_df


def load_production_curves(sheet_name):
    """Read case curves and interpolate onto quarterly dates without extrapolation."""
    raw = pd.read_excel(PRODUCTION_FILE, sheet_name=sheet_name, header=None)
    if raw.shape[1] < 101:
        raise ValueError(f"Unexpected column count in '{sheet_name}': {raw.shape[1]}")

    data = raw.iloc[3:, :].copy()
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    valid = dates.notna()
    dates = dates.loc[valid].dt.normalize()
    values = data.loc[valid].iloc[:, 1:101].apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(dates)
    values = values.groupby(level=0).last().sort_index()

    curves = []
    for case_no in range(1, 101):
        series = values.iloc[:, case_no - 1].dropna().sort_index()
        series = series[~series.index.duplicated(keep="last")]
        if len(series) < 2:
            raise ValueError(f"Case {case_no} has too few values in '{sheet_name}'.")

        # Cumulative production cannot physically decrease. Keep observed
        # cumulative values non-decreasing to remove tiny numerical dips.
        series = series.clip(lower=0).cummax()
        combined = series.index.union(TARGET_DATES).sort_values()
        aligned = (
            series.reindex(combined)
            .interpolate(method="time", limit_area="inside")
            .reindex(TARGET_DATES)
        )
        if aligned.isna().any():
            missing = aligned.index[aligned.isna()].strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"Case {case_no} in '{sheet_name}' cannot be aligned to the full "
                f"1998-2008 quarterly timeline without extrapolation. "
                f"First missing dates: {missing[:5]}"
            )
        curves.append(aligned.to_numpy(float))
    return np.vstack(curves)


def get_split_indices(case_numbers):
    mapping = {int(c): i for i, c in enumerate(case_numbers)}
    def indices(cases):
        absent = [c for c in cases if c not in mapping]
        if absent:
            raise ValueError(f"Cases missing from dataset: {absent}")
        return np.array([mapping[c] for c in cases], dtype=int)
    return indices(TRAIN_CASES), indices(VALIDATION_CASES), indices(TEST_CASES)


def make_model(name):
    if name == "PCA + Gaussian Process":
        return MultiOutputRegressor(
            Pipeline([
                ("scale", StandardScaler()),
                ("gp", GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=1.0)
                    + WhiteKernel(noise_level=1e-5),
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=42,
                )),
            ])
        )
    if name == "PCA + Poly2 Ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if name == "PCA + Neural Network":
        return Pipeline([
            ("scale", StandardScaler()),
            ("nn", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=3000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=80,
                random_state=42,
            )),
        ])
    raise ValueError(name)


def score_metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual.ravel(), predicted.ravel())),
        "RMSE": float(np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel()))),
        "R2": float(r2_score(actual, predicted, multioutput="uniform_average")),
    }


def main():
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(f"Missing production workbook: {PRODUCTION_FILE}")

    cases, X_df = load_case_features()
    feature_df = pd.DataFrame({"Case": cases})
    # Imputation and scaling are fitted ONLY on training rows, preventing
    # information from validation/test cases leaking into preprocessing.
    train_idx, val_idx, test_idx = get_split_indices(cases)

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_df.iloc[train_idx])
    X_val = imputer.transform(X_df.iloc[val_idx])
    X_test = imputer.transform(X_df.iloc[test_idx])

    # Fit scaler on train only. Model pipelines also scale, but scaling the
    # shared inputs here makes feature magnitudes consistent before PCA target
    # modeling and gives an explicit, reproducible preprocessing step.
    # X is NOT PCA-reduced; PCA is applied to the production target curves.
    # (This matches the selected model family: PCA on output curves + regressor.)
    Y_parts = []
    print("\nReading historical production targets...")
    for sheet in PRODUCTION_SHEETS:
        print(f"  {sheet}")
        Y_parts.append(load_production_curves(sheet))
    Y = np.concatenate(Y_parts, axis=1)

    # Standardize target curves and fit PCA only on training cases.
    output_scaler = StandardScaler()
    Y_train_scaled = output_scaler.fit_transform(Y[train_idx])
    Y_val_scaled = output_scaler.transform(Y[val_idx])
    Y_test_scaled = output_scaler.transform(Y[test_idx])

    n_components = min(MAX_PCA_COMPONENTS, len(train_idx) - 1, Y_train_scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(Y_train_scaled)
    Z_val = pca.transform(Y_val_scaled)
    Z_test = pca.transform(Y_test_scaled)

    print("\nDataset summary")
    print(f"  Case-level input shape: {X_df.shape}")
    print(f"  Production target shape: {Y.shape}")
    print(f"  Training cases: {len(train_idx)} (Cases 1-70)")
    print(f"  Validation cases: {len(val_idx)} (Cases 71-85)")
    print(f"  Test cases: {len(test_idx)} (Cases 86-100)")
    print(f"  Quarterly dates per target: {len(TARGET_DATES)}")
    print(f"  PCA components: {n_components}")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    model_names = [
        "PCA + Gaussian Process",
        "PCA + Poly2 Ridge",
        "PCA + Neural Network",
    ]
    metric_rows = []
    pred_store = {}

    for name in model_names:
        print(f"\nTraining {name}...")
        model = make_model(name)
        model.fit(X_train, Z_train)
        z_val_pred = model.predict(X_val)
        z_test_pred = model.predict(X_test)

        y_val_pred = output_scaler.inverse_transform(pca.inverse_transform(z_val_pred))
        y_test_pred = output_scaler.inverse_transform(pca.inverse_transform(z_test_pred))
        pred_store[name] = {"Validation": y_val_pred, "Test": y_test_pred}

        for split, actual, pred in [
            ("Validation", Y[val_idx], y_val_pred),
            ("Test", Y[test_idx], y_test_pred),
        ]:
            row = {"Model": name, "Split": split, **score_metrics(actual, pred)}
            metric_rows.append(row)
            print(f"  {split}: " + ", ".join(f"{k}={v:.5f}" for k, v in row.items() if k not in ["Model", "Split"]))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Separate metrics per production variable, so one target cannot hide
    # poor performance on another target.
    target_metric_rows = []
    n_dates = len(TARGET_DATES)
    for name in model_names:
        for split, case_list, actual_all, pred_all in [
            ("Validation", VALIDATION_CASES, Y[val_idx], pred_store[name]["Validation"]),
            ("Test", TEST_CASES, Y[test_idx], pred_store[name]["Test"]),
        ]:
            for t, label in enumerate(TARGET_LABELS):
                sl = slice(t * n_dates, (t + 1) * n_dates)
                target_metric_rows.append({
                    "Model": name, "Split": split, "Target": label,
                    **score_metrics(actual_all[:, sl], pred_all[:, sl]),
                })
    pd.DataFrame(target_metric_rows).to_csv(
        OUTPUT_DIR / "metrics_by_production_target.csv", index=False
    )

    # Long-form validation/test predictions.
    records = []
    for name in model_names:
        for split, case_list, actual_all, pred_all in [
            ("Validation", VALIDATION_CASES, Y[val_idx], pred_store[name]["Validation"]),
            ("Test", TEST_CASES, Y[test_idx], pred_store[name]["Test"]),
        ]:
            for row_i, case_no in enumerate(case_list):
                for target_i, target_label in enumerate(TARGET_LABELS):
                    sl = slice(target_i * n_dates, (target_i + 1) * n_dates)
                    for j, date in enumerate(TARGET_DATES):
                        records.append({
                            "Model": name, "Split": split, "Case": case_no,
                            "Target": target_label, "Date": date.strftime("%Y-%m-%d"),
                            "Actual": float(actual_all[row_i, sl][j]),
                            "Predicted": float(pred_all[row_i, sl][j]),
                        })
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "validation_test_predictions.csv", index=False)

    pd.DataFrame({
        "Component": np.arange(1, n_components + 1),
        "Variance_Explained": pca.explained_variance_ratio_,
        "Cumulative_Variance_Explained": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(OUTPUT_DIR / "pca_variance_explained.csv", index=False)

    # Comparison plots. For R2 higher is better; MAE/RMSE lower is better.
    for metric_name in ["RMSE", "MAE", "R2"]:
        pivot = metrics_df.pivot(index="Model", columns="Split", values=metric_name)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title(f"Grid-feature Proxy Model Comparison - {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparison_{metric_name.lower()}.png", dpi=200)
        plt.close()

    val_scores = metrics_df[metrics_df["Split"] == "Validation"]
    best = val_scores.loc[val_scores["RMSE"].idxmin()]
    (OUTPUT_DIR / "best_model.txt").write_text(
        f"Best model by validation RMSE: {best['Model']}\n"
        f"Validation RMSE: {best['RMSE']:.8f}\n"
        "Features were summarized from the cleaned 29-column grid-cell case files.\n"
        "Targets are historical cumulative production through 2008 only.\n",
        encoding="utf-8",
    )

    print("\nFINAL COMPARISON")
    print(metrics_df.to_string(index=False))
    print(f"\nSelected by validation RMSE: {best['Model']}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print("\nImportant: this is a historical proxy-model evaluation, not a 2008-2028 forecast.")
    print("Do not delete older results; this run has its own output folder.")


if __name__ == "__main__":
    main()
