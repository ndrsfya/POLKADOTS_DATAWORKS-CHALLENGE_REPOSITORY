"""
History matching only: compare Cases 1–100 with RAMP_History, calculate
Gas/Oil/Water matching metrics, rank cases, and save CSVs and plots.

Required input:
  - 02 Field Production.xlsx

This script does not require ML prediction outputs, a forecast CSV, or the
deliverable template. It does not generate forecasts.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

BASE_DIR = Path(__file__).resolve().parent

# Find required inputs in the project folder or any of its subfolders.
def find_input_file(filename, preferred_paths=()):
    for candidate in preferred_paths:
        if candidate.exists() and candidate.is_file():
            return candidate

    # First check likely locations.
    likely_roots = [
        BASE_DIR,
        BASE_DIR / "Results",
        BASE_DIR / "results",
        BASE_DIR / "Original_Data",
        BASE_DIR / "Cleaned_Data",
        BASE_DIR / "poly2_future_predictions_all_cases",
        BASE_DIR / "poly2_future_predictions_all_cases" / "results",
        BASE_DIR / "poly2_future_predictions_all_cases" / "Results",
    ]
    for root in likely_roots:
        candidate = root / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    # Then recursively search the entire project folder.
    wanted = filename.lower()
    for candidate in BASE_DIR.rglob("*"):
        if candidate.is_file() and candidate.name.lower() == wanted:
            return candidate
    return None


PRODUCTION_FILE = find_input_file(
    "02 Field Production.xlsx",
    [BASE_DIR / "02 Field Production.xlsx",
     BASE_DIR / "Original_Data" / "02 Field Production.xlsx"],
)
OUTPUT_DIR = BASE_DIR / "history_matching_results"
OUTPUT_RANKING = OUTPUT_DIR / "history_match_ranking.csv"
OUTPUT_SCORES = OUTPUT_DIR / "history_match_scores.csv"

PRODUCTION_SHEETS = {
    "Gas": "Gas production cumulative",
    "Oil": "Oil production cumulative",
    "Water": "Water production cumulative",
}
TARGET_TO_TEMPLATE_COL = {
    "Gas": 2,    # column B
    "Oil": 3,    # column C
    "Water": 4,  # column D
}
TOP_N = 10


def read_history_sheet(path, sheet_name):
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    headers = raw.iloc[1].astype(str).str.strip().tolist()
    dates = pd.to_datetime(raw.iloc[3:, 0], errors="coerce")
    valid = dates.notna().to_numpy()
    dates = pd.DatetimeIndex(dates[valid])
    rows = raw.iloc[3:, :].loc[valid]

    result = pd.DataFrame(index=dates)
    result.index.name = "Date"

    for case_num in range(1, 101):
        label = f"Case_{case_num}"
        if label not in headers:
            continue
        col = headers.index(label)
        vals = pd.to_numeric(rows.iloc[:, col], errors="coerce").to_numpy(dtype=float, copy=True)
        vals[vals < 0] = np.nan
        result[label] = vals

    if "RAMP_History" not in headers:
        raise ValueError(f"RAMP_History not found in sheet: {sheet_name}")
    ramp_col = headers.index("RAMP_History")
    ramp = pd.to_numeric(rows.iloc[:, ramp_col], errors="coerce").to_numpy(dtype=float, copy=True)
    ramp[ramp < 0] = np.nan
    result["RAMP_History"] = ramp

    # Average duplicate dates, if any; blanks remain missing, not zero.
    return result.groupby(level=0).mean().sort_index()


def compare_case_to_history(case_series, history_series):
    case_series = case_series.dropna().sort_index()
    history_series = history_series.dropna().sort_index()
    if len(case_series) < 2 or len(history_series) == 0:
        return np.array([]), np.array([]), np.array([])

    # Only compare observed history dates inside the case's available date range.
    mask = (
        (history_series.index >= case_series.index.min())
        & (history_series.index <= case_series.index.max())
    )
    dates = history_series.index[mask]
    actual = history_series.loc[dates].to_numpy(dtype=float)
    if len(dates) == 0:
        return dates, actual, np.array([])

    x_case = case_series.index.view("int64").astype(np.float64)
    x_dates = dates.view("int64").astype(np.float64)
    predicted = np.interp(x_dates, x_case, case_series.to_numpy(dtype=float))
    return dates, actual, predicted


def calc_metrics(actual, predicted):
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = np.asarray(actual, dtype=float)[mask]
    predicted = np.asarray(predicted, dtype=float)[mask]
    n = len(actual)
    if n == 0:
        return {"N": 0, "RMSE": np.nan, "MAE": np.nan, "NRMSE": np.nan, "NMAE": np.nan, "R2": np.nan}

    err = predicted - actual
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    scale = float(np.max(actual) - np.min(actual))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(np.abs(actual)))
    nrmse = rmse / scale if scale > 0 else np.nan
    nmae = mae / scale if scale > 0 else np.nan
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan
    return {"N": n, "RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "NMAE": nmae, "R2": r2}


def history_match(production_file):
    sheet_data = {
        target: read_history_sheet(production_file, sheet)
        for target, sheet in PRODUCTION_SHEETS.items()
    }
    records = []
    aligned = {target: {} for target in PRODUCTION_SHEETS}

    for target, data in sheet_data.items():
        history = data["RAMP_History"].dropna().sort_index()
        for case_num in range(1, 101):
            col = f"Case_{case_num}"
            dates, actual, predicted = compare_case_to_history(data[col], history)
            metrics = calc_metrics(actual, predicted)
            records.append({"Case": case_num, "Target": target, **metrics})
            if len(dates) and len(predicted):
                aligned[target][case_num] = (dates, actual, predicted)

    scores = pd.DataFrame(records)
    wide = scores.pivot(index="Case", columns="Target", values="NRMSE")
    for target in PRODUCTION_SHEETS:
        if target not in wide:
            wide[target] = np.nan

    # Equal-weight average of Gas/Oil/Water normalized RMSE. Require all 3 targets.
    wide["Combined_NRMSE"] = wide[list(PRODUCTION_SHEETS.keys())].mean(axis=1, skipna=False)
    wide["Targets_Scored"] = wide[list(PRODUCTION_SHEETS.keys())].notna().sum(axis=1)
    ranking = wide.reset_index().sort_values(
        ["Combined_NRMSE", "Case"], ascending=[True, True], na_position="last"
    ).reset_index(drop=True)
    ranking["Rank"] = np.arange(1, len(ranking) + 1)
    ranking = ranking[["Rank", "Case", "Gas", "Oil", "Water", "Combined_NRMSE", "Targets_Scored"]]
    return scores, ranking, sheet_data, aligned


def write_history_match_plots(sheet_data, aligned, ranking, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    top_cases = ranking.head(TOP_N)["Case"].astype(int).tolist()
    for target, data in sheet_data.items():
        history = data["RAMP_History"].dropna().sort_index()
        for label, case_list in [("all_100", list(range(1, 101))), ("top10", top_cases)]:
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(history.index, history.values, color="black", linestyle="--",
                    linewidth=2.5, label="RAMP_History (observed)")
            for case_num in case_list:
                if case_num not in aligned[target]:
                    continue
                dates, _, predicted = aligned[target][case_num]
                ax.plot(dates, predicted, linewidth=0.8 if label == "all_100" else 1.5,
                        alpha=0.35 if label == "all_100" else 0.85,
                        label=f"Case {case_num}")
            ax.set_title(f"{label.replace('_', ' ').title()} — {target} cumulative history match")
            ax.set_xlabel("Date")
            ax.set_ylabel(f"{target} cumulative production")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=6 if label == "all_100" else 8, ncol=4 if label == "all_100" else 2)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{label}_{target.lower()}_history_match.png", dpi=150)
            plt.close(fig)



def write_score_ranking_plots(ranking, plot_dir):
    """Save combined-NRMSE bar charts for all cases and the top 20."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    charts = [
        (ranking.copy(), "all_100_combined_nrmse.png",
         "All 100 cases ranked by combined NRMSE"),
        (ranking.head(20).copy(), "top20_combined_nrmse.png",
         "Top 20 historical matches by combined NRMSE"),
    ]

    for frame, filename, title in charts:
        frame = frame.dropna(subset=["Combined_NRMSE"]).sort_values("Combined_NRMSE")
        fig, ax = plt.subplots(figsize=(15, 6))
        labels = [f"Case {int(case)}" for case in frame["Case"]]
        values = frame["Combined_NRMSE"].to_numpy(dtype=float)
        ax.bar(labels, values)
        ax.set_title(title)
        ax.set_xlabel("Original simulation case")
        ax.set_ylabel("Combined NRMSE (lower is better)")
        ax.tick_params(axis="x", rotation=90, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)


def main():
    if PRODUCTION_FILE is None or not PRODUCTION_FILE.exists():
        raise FileNotFoundError(
            "Could not find '02 Field Production.xlsx'. Put it in this project folder "
            f"or a subfolder such as 'Original_Data'. Searched under: {BASE_DIR}"
        )

    print("Located production workbook:")
    print(f"  {PRODUCTION_FILE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\\nStep 1/3: Comparing all 100 simulation cases against RAMP_History...")
    scores, ranking, sheet_data, aligned = history_match(PRODUCTION_FILE)
    scores.to_csv(OUTPUT_SCORES, index=False)
    ranking.to_csv(OUTPUT_RANKING, index=False)

    print("Step 2/3: Creating history-match and ranking plots...")
    plot_dir = OUTPUT_DIR / "plots"
    write_history_match_plots(sheet_data, aligned, ranking, plot_dir)
    write_score_ranking_plots(ranking, plot_dir)

    print("Step 3/3: Finished.")
    print(f"\\nDetailed metrics: {OUTPUT_SCORES}")
    print(f"Full ranking:     {OUTPUT_RANKING}")
    print(f"Plots folder:     {plot_dir}")
    print("\\nTop 10 history-matched cases (lower Combined_NRMSE is better):")
    print(
        ranking.head(TOP_N)[
            ["Rank", "Case", "Gas", "Oil", "Water", "Combined_NRMSE", "Targets_Scored"]
        ].to_string(index=False)
    )
    print("\\nNOTE: This script performs history matching only. It does not use ML results,")
    print("generate forecasts, or populate the final deliverable template.")


if __name__ == "__main__":
    main()
