# Reservoir Production Prediction - DataWorks Challenge 2026

## 1. Project Overview

This project prepares reservoir simulation data, performs history matching, generates production forecasts, evaluates machine learning models, and analyzes the results using OLAP-style operations and 3D visualization.

The main objective is to analyze 100 simulation cases and forecast cumulative gas, oil, and water production from 2008 to 2028, using historical production data available through approximately 2008.

## 2. Project Workflow

1. **Data preparation:** Clean the reservoir grid-cell datasets and prepare the simulation case data.
2. **Proxy-model training:** Train the proxy model using the cleaned simulation-case data with `train_proxy_from_cleaned_cases.py`.
3. **ML prediction:** Use the trained proxy model to generate predictions with `ml_prediction_poly2_ridge.py`.
4. **History matching:** Compare simulated production curves with `RAMP_History` and rank the cases according to their production-history match using `history_matching.py`.
5. **Production forecasting:** Generate future cumulative production estimates through 2028 for all 100 cases using `forecast_production.py`.
6. **OLAP analysis:** Organize forecast data by case, time, and production type to support multidimensional analysis using `olap_analysis.py`.
7. **3D visualization:** Visualize the forecast production curves and differences between simulation cases using `forecast_3d.py`.

## 4. Code Reference and How to Run

| Code / script | How to run in the terminal |
|---|---|
| Data preparation | Open and run `01_Data_Preparation_DAMA.ipynb` in Jupyter. |
| `train_proxy_from_cleaned_cases.py` | `python train_proxy_from_cleaned_cases.py` |
| `ml_prediction_poly2_ridge.py` | `python ml_prediction_poly2_ridge.py` |
| `history_matching.py` | `python history_matching.py` |
| `forecast_production.py` | `python forecast_production.py` |
| `olap_analysis.py` | `python olap_analysis.py` |
| `forecast_3d.py` | `python forecast_3d.py` |

**Important:** Run each script only after its required input files have been generated. The commands assume the scripts are saved in the project root under these exact filenames.

## 5. Main Input Files

| File | Purpose |
|---|---|
| `01 Introduction.xlsx` | Contains uncertainty parameter values for the simulation cases. |
| `02 Field Production.xlsx` | Contains historical production curves, including gas, oil, water, and `RAMP_History`. |
| `03 Train Cases 1.xlsx` to `06 Train Cases 4.xlsx` | Original training-case grid-cell datasets. |
| `07 Validation Cases.xlsx` | Original validation-case dataset. |
| `08 Test Cases.xlsx` | Original test-case dataset. |
| `09 Template Deliverable.xlsx` | Template for the final deliverable, where applicable. |

Cleaned workbooks may use filenames ending in `_Cleaned.xlsx`. Preserve the original files and write cleaned data to separate files.

## 6. Script Descriptions and Outputs

### `history_match.py`

Compares the production curves from the simulation cases with `RAMP_History`. It calculates matching metrics and ranks the cases to identify the best historical matches.

**Main output:** History-matching results, comparison plots, and the top 10 matched cases. Depending on the script configuration, it may also populate the deliverable template using the forecast CSV.

### `forecast_production.py`

Generates and plots cumulative production forecasts for gas, oil, and water from 1998 to 2028. The historical portion should use the available production data, while the forecast portion begins around 2008.

**Main output:** Forecast CSV files and plots for all 100 cases and the selected top 10 cases, depending on the script configuration.

### `ml_prediction.py`

Trains and compares three proxy-model approaches:

- PCA + Gaussian Process
- PCA + Polynomial Regression (degree 2) + Ridge
- PCA + Neural Network

The intended split is Cases 1–70 for training, 71–85 for validation, and 86–100 for testing.

**Main output:** Model evaluation metrics, predicted-versus-actual comparisons, and diagnostic plots.

The historical proxy-model task predicts historical production curves. Its evaluation should not be presented as proof of future forecast accuracy.

### `olap_analysis.py`

Creates an OLAP-style workflow from `poly2_future_predictions_all_cases.csv`. It supports:

- **Slice:** Select a single production type, such as oil.
- **Dice:** Filter selected cases, production types, and years.
- **Roll-up:** Summarize cumulative production at annual intervals.
- **Drill-down:** Inspect detailed production records for an individual case.

**Main output:** The `olap_results` folder, containing the fact table, filtered datasets, annual summaries, and a summary report.

### `forecast_3d.py`

Creates interactive 3D plots from the forecast CSV using Plotly.

**Main output:** The `forecast_3d_results` folder, containing separate HTML visualizations for gas, oil, and water, plus a summary of the final available forecast values.

Open the HTML files in a web browser to rotate, zoom, and explore the graphs.

## 7. Expected Folder Structure

The following is the expected organization of the project files and output folders.

```text
Reservoir_Prediction/
├── Original_Data/
│   ├── 01 Introduction.xlsx
│   ├── 02 Field Production.xlsx
│   ├── 03 Train Cases 1.xlsx
│   ├── 04 Train Cases 2.xlsx
│   ├── 05 Train Cases 3.xlsx
│   ├── 06 Train Cases 4.xlsx
│   ├── 07 Validation Cases.xlsx
│   ├── 08 Test Cases.xlsx
│   └── 09 Template Deliverable.xlsx
├── Raw_Data/
├── Cleaned_Data/
├── proxy_training_grid_features/
├── history_matching_results/
├── ml_prediction_poly2_ridge_results/
├── olap_results/
├── forecast_3d_results/
├── reports/
├── 01_Data_Preparation_DAMA.ipynb
├── train_proxy_from_cleaned_cases.py
├── ml_prediction_poly2_ridge.py
├── history_matching.py
├── forecast_production.py
├── olap_analysis.py
├── forecast_3d.py
└── poly2_future_predictions_all_cases.csv
```

**Notes:**
- `Raw_Data/` contains the original or unprocessed datasets, depending on the data-preparation workflow.
- `Cleaned_Data/` contains the cleaned datasets generated during data preparation.
- `proxy_training_grid_features/` stores proxy-training features or related outputs.
- `history_matching_results/` contains history-matching metrics, rankings, and comparison plots.
- `ml_prediction_poly2_ridge_results/` contains ML prediction results, evaluation metrics, and plots.
- `olap_results/` contains OLAP-style analysis tables and summaries.
- `forecast_3d_results/` contains interactive 3D visualization files.
- `reports/` stores project reports and documentation.
- `poly2_future_predictions_all_cases.csv` contains forecast data used by the OLAP and 3D visualization scripts, if generated by the forecasting workflow.

Output folders and CSV files are created when the corresponding scripts run. Their exact contents depend on the script configuration.

## 8. Recommended Execution Order

1. Run `01_Data_Preparation_DAMA.ipynb` and verify the cleaned datasets.
2. Run `train_proxy_from_cleaned_cases.py` to train the proxy model using the cleaned simulation-case data.
3. Run `ml_prediction_poly2_ridge.py` to generate ML predictions and evaluate the PCA + Polynomial Features (degree 2) + Ridge Regression model.
4. Run `history_matching.py` to compare simulation production against `RAMP_History` and rank the cases.
5. Run `forecast_production.py` to generate the production forecast CSV and plots, after checking the forecasting method and assumptions.
6. Run `olap_analysis.py` to organize and summarize the forecast data.
7. Run `forecast_3d.py` to create interactive 3D forecast visualizations.
8. Review the output files and prepare the final deliverable.

Run each script only after its required input files have been generated. If a script fails, resolve the error before continuing.

Install Python 3 and the required Python packages using the terminal:

```powershell
python -m pip install pandas numpy matplotlib openpyxl scikit-learn plotly
```

Run commands from the project directory:

```powershell
cd "C:\Users\nadee\OneDrive\Desktop\Reservoir_Prediction"
```

The scripts search for their required input files within the project directory where supported. Keep the original workbooks, cleaned data, forecast CSV files, and scripts organized within this directory.

## 9. Important Limitations

- Historical production data is available only through approximately 2008; production from 2008 to 2028 is forecast rather than observed.
- The current polynomial-based future forecast uses assumptions about future production trends. These forecasts are not verified future simulator outputs.
- ML evaluation against assumption-generated targets measures how well the model reproduces those assumptions, not how accurately it predicts actual future reservoir performance.
- OLAP organizes and summarizes data; it does not improve forecast accuracy by itself.
- A 3D production chart is a visualization of production across time and cases. It is not a geological 3D reservoir model.
- Profitability cannot be established from cumulative production alone. Economic evaluation requires additional assumptions or data such as oil and gas prices, operating costs, development costs, and the economic evaluation period.

## 10. Final Objective

The combined workflow provides a structured approach to reservoir data preparation, history matching, production forecasting, machine learning evaluation, multidimensional analysis, and visualization. Together, these outputs support comparison of the 100 simulation cases and identification of promising production scenarios, subject to forecast and data limitations.
