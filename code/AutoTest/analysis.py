import os
import ast
import re
import pandas as pd
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

AUTO_TEST_PATH = "/home/pavel/AutoTest/code/AutoTest"
SDC_DIR = '/home/pavel/Datasets/SDCs'
DATA_PATH = '/home/pavel/Datasets/ToScan/'

SDCs = [
    "tablib_selected_sdc.csv",
    "rt_train_selected_sdc.csv",
    "st_train_selected_sdc.csv"
]

Datasets = [
    ["rein-datasets_adult_clean.csv", "rein-datasets_adult_dirty.csv"],
    ["Quintet_beers_clean.csv", "Quintet_beers_dirty.csv"],
    ["Quintet_flights_clean.csv", "Quintet_flights_dirty.csv"],
    ["Quintet_hospital_clean.csv", "Quintet_hospital_dirty.csv"],
    ["Quintet_movies_1_clean.csv", "Quintet_movies_1_dirty.csv"],
    ["Quintet_rayyan_clean.csv", "Quintet_rayyan_dirty.csv"],
    ["soccer_clean.csv", "soccer_dirty.csv"],
    ["tax_clean.csv", "tax_dirty.csv"],
    ["Benchmarks_rt_bench_clean.csv", "Benchmarks_rt_bench_dirty.csv"],
    ["Benchmarks_st_bench_clean.csv", "Benchmarks_st_bench_dirty.csv"]
]

METRICS = ['TP', 'FP', 'TN', 'FN', '% Errors', 'Precision', 'Recall', 'F1_Score', 'runtime']

# ── Helpers ───────────────────────────────────────────────────────────────────

def fname(path):
    return os.path.splitext(os.path.basename(path))[0]


def dataset_label(dirty_filename):
    """Strip common prefixes and '_dirty' suffix for a clean column label."""
    name = fname(dirty_filename)
    for prefix in ('Benchmarks_', 'Quintet_', 'rein-datasets_'):
        name = name.replace(prefix, '')
    name = name.replace('_dirty', '').replace('_bench', '')
    return name


def method_label(sdc_filename):
    """Extract method name from SDC filename."""
    return fname(sdc_filename).replace('_selected_sdc', '')


def format_runtime(seconds):
    """Convert runtime in seconds to hh:mm:ss:ms format."""
    if seconds is None:
        return None
    try:
        seconds = float(seconds)
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        secs = int(remaining % 60)
        milliseconds = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{milliseconds:03d}"
    except (ValueError, TypeError):
        return None


def parse_runtime(metadata_path):
    """Read execution time (in seconds) from a metadata file."""
    try:
        with open(metadata_path, 'r') as f:
            content = f.read()
        match = re.search(r'execution time:\s*([\d.]+)', content)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


# ── Core error matrix builder ─────────────────────────────────────────────────

def build_error_matrices(clean_path, dirty_path, output_df):
    """
    Build actual_errors and detected_errors boolean DataFrames.
    Shared by both the summary and per-column metric functions.

    Returns: (dirty_df, actual_errors, detected_errors)
    """
    clean_df = pd.read_csv(clean_path, dtype=str)
    dirty_df = pd.read_csv(dirty_path, dtype=str)

    if clean_df.shape != dirty_df.shape:
        raise ValueError(f"Shape mismatch. Clean: {clean_df.shape}, Dirty: {dirty_df.shape}")

    actual_errors = (clean_df.fillna('') != dirty_df.fillna(''))
    detected_errors = pd.DataFrame(False, index=dirty_df.index, columns=dirty_df.columns)

    if not output_df.empty:
        for _, row in output_df.iterrows():
            if 'header' not in row or 'outlier' not in row:
                continue

            column_name = row['header']
            error_values = row['outlier']

            if pd.isna(column_name) or pd.isna(error_values):
                continue

            # Parse the list of outliers from the string representation
            if isinstance(error_values, str):
                try:
                    error_values = ast.literal_eval(error_values)
                except (ValueError, SyntaxError):
                    error_values = [error_values]  # fallback: treat as single value
            elif not isinstance(error_values, list):
                error_values = [error_values]

            if column_name in dirty_df.columns:
                for error_value in error_values:
                    mask = dirty_df[column_name].fillna('') == str(error_value)
                    detected_errors.loc[mask, column_name] = True
            else:
                print(f"Warning: Column '{column_name}' not found in data files.")

    return dirty_df, actual_errors, detected_errors


# ── Metrics calculation ───────────────────────────────────────────────────────

def calculate_error_detection_metrics(clean_path, dirty_path, output_df):
    """
    Calculate TP, FP, FN, TN and derived metrics for an error detection run.

    Returns:
    --------
    dict : TP, FP, FN, TN, Total_Cells, % Errors, Precision, Recall, F1_Score
    """
    _, actual_errors, detected_errors = build_error_matrices(clean_path, dirty_path, output_df)

    TP = int(((actual_errors) & (detected_errors)).sum().sum())
    FP = int(((~actual_errors) & (detected_errors)).sum().sum())
    FN = int(((actual_errors) & (~detected_errors)).sum().sum())
    TN = int(((~actual_errors) & (~detected_errors)).sum().sum())
    total_cells = TP + FP + FN + TN

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
    pct_errors = round(((TP + FN) / total_cells) * 100, 2) if total_cells > 0 else 0

    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'Total_Cells': total_cells,
        '% Errors': pct_errors,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'F1_Score': round(f1_score, 4),
    }


def calculate_per_column_metrics(clean_path, dirty_path, output_df):
    """
    Calculate TP, FP, TN, FN and % Errors per column.
    Includes all columns, even those with no detected errors.
    % Errors = (TP + FN) / (TP + FP + TN + FN)

    Returns:
    --------
    pd.DataFrame : rows = [TP, FP, TN, FN, % Errors], columns = data columns
                   or None if the dataset has more than 20 columns.
    """
    _, actual_errors, detected_errors = build_error_matrices(clean_path, dirty_path, output_df)

    if len(actual_errors.columns) > 20:
        return None

    results = {}
    for col in actual_errors.columns:
        TP = int((actual_errors[col] & detected_errors[col]).sum())
        FP = int((~actual_errors[col] & detected_errors[col]).sum())
        TN = int((~actual_errors[col] & ~detected_errors[col]).sum())
        FN = int((actual_errors[col] & ~detected_errors[col]).sum())
        total = TP + FP + TN + FN
        pct_errors = round(((TP + FN) / total) * 100, 2) if total > 0 else 0
        results[col] = {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN, '% Errors': pct_errors}

    # rows = metrics, columns = data columns
    return pd.DataFrame(results).loc[['TP', 'FP', 'TN', 'FN', '% Errors']]


# ── Table building ────────────────────────────────────────────────────────────

def build_tables():
    """
    Compute summary metrics for all available result CSVs.
    Returns a dict of DataFrames keyed by method name.
    Safe to run while experiment.py is still running.
    """
    result_dir = f"{AUTO_TEST_PATH}/results/detected_outliers"
    table_data = defaultdict(lambda: defaultdict(dict))  # [method][metric][dataset]

    for sdc in SDCs:
        method = method_label(sdc)
        sdc_path = f"{SDC_DIR}/{sdc}"

        for clean, dirty in Datasets:
            dataset = dataset_label(dirty)
            result_path = f"{result_dir}/{fname(sdc_path)}_on_{fname(dirty)}.csv"
            metadata_path = f"{result_dir}/runtime_{fname(sdc_path)}_on_{fname(dirty)}.csv"

            if not os.path.exists(result_path):
                continue  # not yet scanned, skip silently

            try:
                output_df = pd.read_csv(result_path)
                res = calculate_error_detection_metrics(
                    f"{DATA_PATH}/{clean}",
                    f"{DATA_PATH}/{dirty}",
                    output_df
                )
                for metric in METRICS:
                    if metric == 'runtime':
                        runtime_seconds = parse_runtime(metadata_path)
                        table_data[method]['runtime'][dataset] = format_runtime(runtime_seconds)
                    else:
                        table_data[method][metric][dataset] = res[metric]
            except Exception as e:
                print(f"Error analysing {method} / {dataset}: {e}")
                for metric in METRICS:
                    table_data[method][metric][dataset] = 'ERROR'

    # Convert to DataFrames: rows = metrics, columns = datasets
    tables = {}
    for method in [method_label(s) for s in SDCs]:
        if method not in table_data:
            continue
        df = pd.DataFrame(table_data[method]).T
        df = df[sorted(df.columns)]
        df = df.loc[[m for m in METRICS if m in df.index]]
        tables[method] = df

    return tables


def build_per_column_tables():
    """
    Compute per-column metrics for all available result CSVs.
    Returns a nested dict: [dataset][method] -> DataFrame
    Safe to run while experiment.py is still running.
    """
    result_dir = f"{AUTO_TEST_PATH}/results/detected_outliers"
    per_column = defaultdict(dict)  # [dataset][method] -> DataFrame

    for sdc in SDCs:
        method = method_label(sdc)
        sdc_path = f"{SDC_DIR}/{sdc}"

        for clean, dirty in Datasets:
            dataset = dataset_label(dirty)
            result_path = f"{result_dir}/{fname(sdc_path)}_on_{fname(dirty)}.csv"

            try:
                # Use empty output_df if not yet scanned — gives % Errors only, TP/FP/FN = 0
                output_df = pd.read_csv(result_path) if os.path.exists(result_path) else pd.DataFrame()
                df = calculate_per_column_metrics(
                    f"{DATA_PATH}/{clean}",
                    f"{DATA_PATH}/{dirty}",
                    output_df
                )
                if df is not None:
                    per_column[dataset][method] = df
            except Exception as e:
                print(f"Error in per-column analysis for {method} / {dataset}: {e}")

    return per_column


# ── Markdown output ───────────────────────────────────────────────────────────

def tables_to_markdown(tables):
    markdown = "# Autotest Experiment Results\n\n"
    for method in [method_label(s) for s in SDCs]:
        if method in tables:
            markdown += f"## {method}\n\n"
            markdown += tables[method].to_markdown()
            markdown += "\n\n"
    return markdown


def per_column_tables_to_markdown(per_column):
    markdown = "# Per-Column Error Breakdown\n\n"
    for dataset in sorted(per_column.keys()):
        for method in [method_label(s) for s in SDCs]:
            if method in per_column[dataset]:
                df = per_column[dataset][method]
                markdown += f"## {dataset} — {method}\n\n"
                markdown += df.to_markdown()
                markdown += "\n\n"
    return markdown


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tables = build_tables()
    print(tables_to_markdown(tables))

    per_column = build_per_column_tables()
    print(per_column_tables_to_markdown(per_column))