import os
import sys
import tempfile
import pandas as pd
import subprocess
import time
import datetime
import traceback

from matplotlib import table
from sqlalchemy.sql.functions import current_timestamp

AUTO_TEST_PATH = "/raid0_ssd2/pavel/AutoTest/code/AutoTest"
SDC_DIR = '/raid0_ssd2/pavel/AutoTest/code/AutoTest/results/SDC'
DATA_PATH = '/raid0_ssd2/pavel/AutoMatelda/datasets'

SDCs = [
    #"tablib_selected_sdc.csv",
    #"rt_train_selected_sdc.csv",
    #"st_train_selected_sdc.csv"
    "combined_sdc.csv"
]


def run_auto_test(sdc_path, csv_path):
    """
    Use AutoTest to apply the SDC to the CSV and measure the execution time.
    """
    start_time = time.time()
    result = subprocess.run(
        ['conda', 'run', '-n', 'VENV', 'python3', './online_detect.py', csv_path, sdc_path],
        text=True,
        capture_output=True,
        cwd=AUTO_TEST_PATH
    )
    execution_time = time.time() - start_time
    if result.returncode != 0:
        print(f"returncode {result.returncode}, Error running Auto-Test: {result.stderr}")
        raise Exception("result.stderr")
    output_path = f"{AUTO_TEST_PATH}/results/detected_outliers/{fname(sdc_path)}_on_{fname(csv_path)}.csv"
    return result, output_path, execution_time


def fname(path):
    return os.path.splitext(os.path.basename(path))[0]

def clean_csv(src, dest):
    """
    Removes values that AutoTest cannot handle and saves the resulting CSV to dest.
    """
    df = pd.read_csv(src, dtype="str")
    #df.dropna(how='all', axis=1, inplace=True)
    df.to_csv(dest)

def load(src):
    df = pd.read_table(src, dtype="str")
    return df

def scan_csv(sdc_path, csv_path, dataset, alias=None, rerun=False):
    """
    Runs AutoTest on csv_path. Saves result to the AutoTest results folder.
    alias: optional name to use in output filenames instead of the csv filename.
    Returns result df and execution time.
    """
    result_dir = f"{AUTO_TEST_PATH}/results/detected_outliers"
    csv_name = alias if alias else fname(csv_path)
    final_path = f"{result_dir}/{fname(sdc_path)}_on_{csv_name}.csv"
    metadata_dir = f"{result_dir}/{csv_name}"

    if os.path.exists(final_path) and not rerun:
        print(f"Already scanned {csv_name}")
        return pd.read_table(final_path), 0.0

    with tempfile.TemporaryDirectory() as temp_dir_name:
        # AutoTest uses the filename to name its output, so we name the temp
        # file after the alias so the output path stays consistent.
        temp_csv_name = f"{csv_name}.csv"
        temp_path = os.path.join(temp_dir_name, temp_csv_name)
        clean_csv(csv_path, temp_path)
        result, output_path, execution_time = run_auto_test(sdc_path, temp_path)

    time_csv_path = os.path.join(result_dir, "time.csv")
    time_df = pd.DataFrame(columns=["Timestamp","dataset","csv","sdc","execution_time","returncode"])

    if not os.path.exists(time_csv_path):
        time_df.to_csv(time_csv_path, index=True, header=True)
    time_df = pd.read_csv(time_csv_path, index_col=0)
    time_df.loc[len(time_df)] = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  dataset,
                                  csv_name,
                                  fname(sdc_path),
                                  execution_time,
                                  result.returncode]
    time_df.to_csv(time_csv_path, index=True, header=True)

    if not os.path.exists(output_path):
        print(f"No errors detected for {csv_name}")
        df = pd.DataFrame(columns=['header', 'outlier'])
        os.makedirs(result_dir, exist_ok=True)
        df.to_csv(final_path, index=False)
        return df, execution_time

    df = load(output_path)
    return df, execution_time


def experiment(dataset):
    """
    Scans all dirty.csv files in subfolders of DGOV_TYPO_PATH.
    Uses the subfolder name as the dataset identifier to avoid filename collisions.
    """
    print(1)
    dataset_path = os.path.join(DATA_PATH, dataset)
    table_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])

    print(table_dirs)

    for sdc in SDCs:
        sdc_path = f"{SDC_DIR}/{sdc}"
        for table_dir in table_dirs:
            dirty_path = os.path.join(dataset_path, table_dir, "dirty.csv")
            if not os.path.exists(dirty_path):
                print(f"No dirty.csv found in {table_dir}, skipping.")
                continue
            print(f"Running {sdc} on {table_dir}")
            try:
                # Use the folder name as the alias so results are named
                # e.g. rt_train_selected_sdc_on_305b_Assessed_Lake_2020.csv
                scan_csv(sdc_path, dirty_path, dataset, alias=table_dir, rerun=False)

            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "synthetic_tables"

    print(f"running experiment on {dataset}")
    experiment(dataset)