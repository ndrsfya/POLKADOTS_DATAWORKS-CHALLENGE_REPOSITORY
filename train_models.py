# train_models.py
# Trains and compares:
# 1. PCA + Gaussian Process
# 2. PCA + Polynomial Degree 2 + Ridge
# 3. PCA + Neural Network
#
# Expected folder structure:
# Reservoir_Prediction/
#   train_models.py
#   Original_Data/
#     01 Introduction.xlsx
#     02 Field Production.xlsx
#   Cleaned_Data/
#     six cleaned simulation workbooks
#   09 Template Deliverable.xlsx
#
# Note: this version reads the two original workbooks only. It does not yet
# use the six Cleaned_Data workbooks because their 29 grid-cell columns are
# cell-level properties, not automatically one row of case-level model inputs.
#
# Production workbook structure confirmed from the provided file:
#   - Sheets: Gas production cumulative, Oil production cumulative,
#     Water production cumulative
#   - Row 1: X, Y1...Y100
#   - Row 2: Date, Case_1...Case_100
#   - Row 3: units
#   - Data begins on row 4
#
# Cases: Train 1-70, validation 71-85, test 86-100.
# Source workbooks are read only and are not modified.

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = Path(__file__).resolve().parent
PARAMETER_FILE = BASE_DIR / "Original_Data" / "01 Introduction.xlsx"
PRODUCTION_FILE = BASE_DIR / "Original_Data" / "02 Field Production.xlsx"
OUTPUT_DIR = BASE_DIR / "model_training_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PARAMETER_SHEET = "Uncertainty Values"
PRODUCTION_SHEETS = [
    "Gas production cumulative",
    "Oil production cumulative",
    "Water production cumulative",
]
TRAIN_CASES = list(range(1, 71))
VALIDATION_CASES = list(range(71, 86))
TEST_CASES = list(range(86, 101))

# Use up to 20 PCA components. This is limited by the number of training cases.
MAX_PCA_COMPONENTS = 20


def load_parameters():
    """Read the four uncertain parameters for Case 1 through Case 100."""
    df = pd.read_excel(PARAMETER_FILE, sheet_name=PARAMETER_SHEET)
    df = df.dropna(how="all").dropna(axis=1, how="all")

    # In the supplied file, the first column is Item and rows are Case 1...Case 100.
    item_col = df.columns[0]
    df["_case"] = (
        df[item_col].astype(str).str.extract(r"(\d+)")[0].astype(float)
    )
    df = df[df["_case"].between(1, 100)].copy()

    expected_parameters = [
        "Fault Transmissibility",
        "Porosity Multiplier",
        "Permeability Multiplier",
        "Aquifer Pore Volume",
    ]
    missing = [c for c in expected_parameters if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing expected input parameter columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df[["_case"] + expected_parameters].sort_values("_case")
    if df["_case"].nunique() != 100:
        raise ValueError("Expected exactly 100 cases in the Uncertainty Values sheet.")

    X = df[expected_parameters].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(X).all():
        raise ValueError("Missing or invalid values found in uncertain parameters.")

    print("Input features:")
    for col in expected_parameters:
        print(f"  - {col}")
    return df["_case"].to_numpy(int), X


def load_production_curve(sheet_name, case_numbers):
    """
    Read the 100 case-specific production curves and align them to a common
    quarterly timeline. The source workbook contains sparse event/timestep
    values, so blanks are not automatically treated as zero.

    Interpolation is performed only between recorded values. No extrapolation
    is performed outside each case's recorded date range.
    """
    raw = pd.read_excel(
        PRODUCTION_FILE,
        sheet_name=sheet_name,
        header=None
    )

    # Supplied workbook layout:
    # row 0 = Y1...Y101, row 1 = Date / Case_1...Case_100 / RAMP_History,
    # row 2 = units, row 3 onward = data.
    if raw.shape[1] < 101 or len(raw) < 4:
        raise ValueError(
            f"Unexpected layout in '{sheet_name}': shape={raw.shape}. "
            "Expected Date + 100 case columns and data beginning on Excel row 4."
        )

    data = raw.iloc[3:, :].copy()
    dates = pd.to_datetime(data.iloc[:, 0], errors="coerce")
    valid_date_rows = dates.notna()
    data = data.loc[valid_date_rows].copy()
    dates = dates.loc[valid_date_rows].dt.normalize()

    # Keep columns B: CW (Case_1 ... Case_100); exclude final RAMP_History column.
    case_values = data.iloc[:, 1:101].copy()
    case_values = case_values.apply(pd.to_numeric, errors="coerce")
    case_values.index = pd.DatetimeIndex(dates)

    # Combine duplicate dates, if any, by taking the last recorded value.
    case_values = case_values.groupby(level=0).last().sort_index()

    # Use common quarterly dates from 1998-01-01 through 2008-01-01.
    # The source data cover this historical interval.
    common_dates = pd.date_range(
        start="1998-01-01",
        end="2008-01-01",
        freq="QS"
    )

    aligned_curves = []
    for case_no in case_numbers:
        # case_values columns are in case order: Case_1 ... Case_100.
        series = case_values.iloc[:, int(case_no) - 1].dropna().sort_index()

        if series.empty:
            raise ValueError(
                f"No numeric production values found for Case {case_no} "
                f"in '{sheet_name}'."
            )

        # Reindex to include target dates, then interpolate only between
        # observed values. limit_area='inside' prevents extrapolation.
        combined_index = series.index.union(common_dates)
        interpolated = (
            series.reindex(combined_index)
            .sort_index()
            .interpolate(method="time", limit_area="inside")
            .reindex(common_dates)
        )

        if interpolated.isna().any():
            missing_dates = common_dates[interpolated.isna()].strftime("%Y-%m-%d").tolist()
            raise ValueError(
                f"Case {case_no} in '{sheet_name}' cannot be aligned to all "
                f"quarterly dates without extrapolation. Missing dates: {missing_dates[:5]}. "
                "Inspect that case's recorded date range before training."
            )

        aligned_curves.append(interpolated.to_numpy(dtype=float))

    values = np.vstack(aligned_curves)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values remain after alignment in '{sheet_name}'.")

    return pd.Series(common_dates), values


def metrics(actual, predicted):
    return {
        "MAE": mean_absolute_error(actual.ravel(), predicted.ravel()),
        "RMSE": np.sqrt(mean_squared_error(actual.ravel(), predicted.ravel())),
        "R2": r2_score(actual, predicted, multioutput="uniform_average"),
    }


def make_regressor(model_name):
    """Regress from four input parameters to PCA coefficients."""
    if model_name == "PCA + Gaussian Process":
        return MultiOutputRegressor(
            Pipeline([
                ("x_scaler", StandardScaler()),
                ("gp", GaussianProcessRegressor(
                    kernel=(
                        ConstantKernel(1.0)
                        * RBF(length_scale=1.0)
                        + WhiteKernel(noise_level=1e-5)
                    ),
                    normalize_y=True,
                    n_restarts_optimizer=1,
                    random_state=42,
                )),
            ])
        )

    if model_name == "PCA + Poly2 Ridge":
        return Pipeline([
            ("x_scaler", StandardScaler()),
            ("poly2", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0)),
        ])

    if model_name == "PCA + Neural Network":
        return Pipeline([
            ("x_scaler", StandardScaler()),
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

    raise ValueError(f"Unknown model: {model_name}")


def main():
    if not PARAMETER_FILE.exists():
        raise FileNotFoundError(f"Cannot find {PARAMETER_FILE.name} in {BASE_DIR}")
    if not PRODUCTION_FILE.exists():
        raise FileNotFoundError(f"Cannot find {PRODUCTION_FILE.name} in {BASE_DIR}")

    case_numbers, X = load_parameters()

    # Load the three cumulative production targets.
    target_curves = []
    common_dates = None
    for sheet in PRODUCTION_SHEETS:
        print(f"\nReading production sheet: {sheet}")
        dates, curves = load_production_curve(sheet, case_numbers)
        if common_dates is None:
            common_dates = dates
        elif not common_dates.equals(dates):
            raise ValueError(f"Date rows differ in production sheet '{sheet}'.")
        target_curves.append(curves)

    # Each case's target vector is [all gas dates, all oil dates, all water dates].
    Y = np.concatenate(target_curves, axis=1)

    case_to_row = {case: i for i, case in enumerate(case_numbers)}
    def indices(cases):
        missing = [case for case in cases if case not in case_to_row]
        if missing:
            raise ValueError(f"Missing case numbers: {missing}")
        return np.array([case_to_row[case] for case in cases], dtype=int)

    train_idx = indices(TRAIN_CASES)
    val_idx = indices(VALIDATION_CASES)
    test_idx = indices(TEST_CASES)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

    # Standardize output curves and fit PCA only on training cases.
    output_scaler = StandardScaler()
    Y_train_scaled = output_scaler.fit_transform(Y_train)
    Y_val_scaled = output_scaler.transform(Y_val)
    Y_test_scaled = output_scaler.transform(Y_test)

    n_components = min(
        MAX_PCA_COMPONENTS,
        Y_train_scaled.shape[0] - 1,
        Y_train_scaled.shape[1],
    )
    pca = PCA(n_components=n_components, random_state=42)
    Z_train = pca.fit_transform(Y_train_scaled)
    Z_val = pca.transform(Y_val_scaled)
    Z_test = pca.transform(Y_test_scaled)

    print("\nData summary")
    print(f"  Cases: {len(case_numbers)}")
    print(f"  Dates per target: {len(common_dates)}")
    print(f"  Combined target shape: {Y.shape}")
    print(f"  PCA components: {n_components}")
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.4f}")

    model_names = [
        "PCA + Gaussian Process",
        "PCA + Poly2 Ridge",
        "PCA + Neural Network",
    ]
    metric_rows = []
    predictions = {}

    for name in model_names:
        print(f"\n{'=' * 60}\nTraining {name}\n{'=' * 60}")
        model = make_regressor(name)
        model.fit(X_train, Z_train)

        Z_val_pred = model.predict(X_val)
        Z_test_pred = model.predict(X_test)

        Y_val_pred = output_scaler.inverse_transform(pca.inverse_transform(Z_val_pred))
        Y_test_pred = output_scaler.inverse_transform(pca.inverse_transform(Z_test_pred))

        predictions[name] = {"Validation": Y_val_pred, "Test": Y_test_pred}

        for split, actual, predicted in [
            ("Validation", Y_val, Y_val_pred),
            ("Test", Y_test, Y_test_pred),
        ]:
            score = metrics(actual, predicted)
            metric_rows.append({"Model": name, "Split": split, **score})
            print(split, {k: round(v, 5) for k, v in score.items()})

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # Save predictions as a long CSV for graphing and later use.
    target_names = [
        "Gas production cumulative",
        "Oil production cumulative",
        "Water production cumulative",
    ]
    n_dates = len(common_dates)
    records = []
    for name in model_names:
        for split, cases, actual, predicted in [
            ("Validation", VALIDATION_CASES, Y_val, predictions[name]["Validation"]),
            ("Test", TEST_CASES, Y_test, predictions[name]["Test"]),
        ]:
            for i, case in enumerate(cases):
                for target_i, target_name in enumerate(target_names):
                    offset = target_i * n_dates
                    for j, date in enumerate(common_dates):
                        records.append({
                            "Model": name,
                            "Split": split,
                            "Case": case,
                            "Target": target_name,
                            "Date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                            "Actual": float(actual[i, offset + j]),
                            "Predicted": float(predicted[i, offset + j]),
                        })
    pd.DataFrame(records).to_csv(
        OUTPUT_DIR / "validation_test_predictions.csv", index=False
    )

    pd.DataFrame({
        "Component": np.arange(1, n_components + 1),
        "Variance_Explained": pca.explained_variance_ratio_,
        "Cumulative_Variance_Explained": np.cumsum(pca.explained_variance_ratio_),
    }).to_csv(OUTPUT_DIR / "pca_variance_explained.csv", index=False)

    # Comparison plots.
    for metric_name in ["RMSE", "MAE", "R2"]:
        pivot = metrics_df.pivot(index="Model", columns="Split", values=metric_name)
        ax = pivot.plot(kind="bar", figsize=(10, 5))
        ax.set_title(f"Model Comparison - {metric_name}")
        ax.set_ylabel(metric_name)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=15)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"comparison_{metric_name.lower()}.png", dpi=200)
        plt.close()

    # Select using validation RMSE, not test RMSE.
    val_scores = metrics_df[metrics_df["Split"] == "Validation"]
    best = val_scores.loc[val_scores["RMSE"].idxmin()]
    (OUTPUT_DIR / "best_model.txt").write_text(
        f"Best model by validation RMSE: {best['Model']}\n"
        f"Validation RMSE: {best['RMSE']:.8f}\n"
        "Selection is based on validation performance.\n",
        encoding="utf-8",
    )

    print("\nFINAL METRICS")
    print(metrics_df.to_string(index=False))
    print(f"\nBest model by validation RMSE: {best['Model']}")
    print(f"Results saved to: {OUTPUT_DIR.resolve()}")
    print(
        "\nNOTE: This trains and validates production curves over the dates present "
        "in 02 Field Production.xlsx (1998-2008). It does not create new 2008-2028 "
        "simulation targets by itself."
    )


if __name__ == "__main__":
    main()
