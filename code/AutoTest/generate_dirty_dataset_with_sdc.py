import warnings
warnings.filterwarnings("ignore")
import argparse
import os

# Disable tokenizers parallelism before loading anything heavy
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass

import ast
import pandas as pd
from config import config
from util import utils, sbert_utils, doduo_utils


parser = argparse.ArgumentParser()
parser.add_argument("src_dataset", help="Path to the source dataset folder.")
parser.add_argument("dst_dataset", help="Path to output the mutated dataset folder.")
parser.add_argument("sdc_fname", help="Path to the file of learned SDCs.")
parser.add_argument("--error_rate", type=float, default=0.05, help="Percentage of cells to corrupt.")
args = parser.parse_args()

sdc_fname = os.path.splitext(os.path.basename(args.sdc_fname))[0]

n_cores = getattr(config, 'n_cores', float('inf'))
cpu_count = os.cpu_count() or 1
num_procs = max(1, cpu_count // 4) if n_cores == float('inf') else max(1, min(int(n_cores), cpu_count) // 4)

# 1. Pre-load all tables and gather a global candidate pool
print("Building global candidate pool and reading tables...")
all_tables_data = {}
global_candidate_pool = set()

if not os.path.exists(args.dst_dataset):
    os.makedirs(args.dst_dataset)

for table in os.listdir(args.src_dataset):
    csv_path = os.path.join(args.src_dataset, table, "clean.csv")
    if not os.path.exists(csv_path): continue
    input_df = pd.read_csv(csv_path, dtype=str)
    all_tables_data[table] = input_df
    for col in input_df.columns:
        global_candidate_pool.update(input_df[col].dropna().unique())

global_candidate_pool = list(global_candidate_pool)
print(f"Global candidate pool built with {len(global_candidate_pool)} unique values across {len(all_tables_data)} tables.")

import random
ground_truth_records = []

for table, input_df in all_tables_data.items():
    csv_fname = table
    print(f"\n--- Processing Table: {csv_fname} ---")

    df = input_df.apply(lambda x: [list([v for v in x.tolist() if pd.notna(v)])], axis=0).T.reset_index()
    df.columns = ['header', 'dist_val']

    rule_df = pd.read_csv(args.sdc_fname, sep='\t')
    rule_list = rule_df['SDC'].apply(ast.literal_eval).to_list()

    sbert_dist_val_embeddings = None
    doduo_dist_val_scores = None

    if any([rule[0][0] == 'sbert' for rule in rule_list]):
        print(f"Computing SentenceBERT embeddings for {csv_fname}")
        sbert_dist_val_embeddings = sbert_utils.dist_val_embeddings_parallel(df, n_proc = num_procs)

    if any([rule[0][0] == 'doduo' for rule in rule_list]):
        print(f"Computing Doduo preprocessing results for {csv_fname}")
        doduo_intermediate_result_dir = os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo)
        doduo_dist_val_scores_fname = f'{csv_fname}_dist_val_scores.pickle'
        doduo_utils.dist_val_scores_parallel(df, doduo_intermediate_result_dir, doduo_dist_val_scores_fname, n_proc = num_procs)
        doduo_dist_val_scores = pd.read_pickle(str(os.path.join(config.dir.storage_root_dir, config.dir.storage_root.doduo, doduo_dist_val_scores_fname)))

    pre_list = list(set([r[0] for r in rule_list]))
    test_matching_dict = utils.build_matching_idx_dict_from_pre_list_parallel(df, pre_list, n_proc=num_procs,
                                                                              sbert_dist_val_embeddings=sbert_dist_val_embeddings,
                                                                              doduo_dist_val_scores=doduo_dist_val_scores)

    # Identify which columns fulfill which rules
    applicable_constraints = []
    # test_matching_dict keys are pre-conditions, values are lists of df row indices (columns)
    for rule in rule_list:
        pre = tuple(rule[0])
        if pre in test_matching_dict and len(test_matching_dict[pre]) > 0:
            for col_idx in test_matching_dict[pre]:
                applicable_constraints.append({'col_idx': col_idx, 'rule': rule})

    if not applicable_constraints:
        print(f"No SDC pre-conditions matched for {csv_fname}. Skipping corruption for this table.")
        # Alternatively, inject random data, but skipping maintains SDC-only errors.
        dst_folder = os.path.join(args.dst_dataset, table)
        if not os.path.exists(dst_folder): os.makedirs(dst_folder)
        input_df.to_csv(str(os.path.join(dst_folder, "clean.csv")), index=False)
        continue

    # Calculate number of cells to corrupt
    total_cells = input_df.size
    num_errors = max(1, int(total_cells * args.error_rate))

    print(f"Targeting {num_errors} errors for {csv_fname}...")

    dirty_df = input_df.copy()

    for i in range(num_errors):
        # 1. Randomly pick a pre-condition constraint targeting a column
        target = random.choice(applicable_constraints)
        col_idx = target['col_idx']
        target_rule = target['rule']
        col_header = df.loc[col_idx, 'header']

        pre, constraint, cohenh, conf, contingency = target_rule
        rule_type = constraint[0]

        # 2. Find a violator value from the candidate pool
        violator = None
        attempts = 0
        while violator is None and attempts < 10:
            attempts += 1
            batch = random.sample(global_candidate_pool, min(100, len(global_candidate_pool)))

            if rule_type == 'pattern':
                pattern = pre[3]
                import regex as re
                for val in batch:
                    if re.match(pattern, val) is None:
                        violator = val
                        break
            elif rule_type == 'pyfunc':
                from check import pyfunc_check
                # Check pyfunc outliers
                outliers = pyfunc_check.get_all_outliers(batch, constraint[1])
                if outliers: violator = outliers[0]
            elif rule_type == 'validator':
                from check import validator_check
                outliers = validator_check.get_all_outliers(batch, constraint[1])
                if outliers: violator = outliers[0]
            elif rule_type == 'sbert':
                from check import sbert_check
                from util import sbert_utils
                ref_embed = sbert_utils.decide_embedding(pre[2])
                dist_thres = constraint[1]
                batch_embeddings = sbert_utils.dist_val_embeddings(batch)
                outliers = sbert_check.get_all_outliers(batch, batch_embeddings, ref_embed, dist_thres)
                if outliers: violator = outliers[0]
            elif rule_type == 'doduo':
                # Simplified check for doduo - assume random sample fails
                violator = random.choice(batch)
            elif rule_type == 'cta':
                violator = random.choice(batch)
            elif rule_type == 'embed':
                from check import embed_check
                dist_thres = constraint[1]
                ref_emb = pre[2]
                outliers = embed_check.get_all_outliers(batch, ref_emb, dist_thres)
                if outliers: violator = outliers[0]

            if violator is not None:
                break

        if violator is None:
            # Fallback if no specific violator found in 1000 sampled
            violator = "SDC_FALLBACK_CORRUPTION_" + str(random.randint(1000, 9999))

        # 3. Inject
        valid_rows = dirty_df[dirty_df[col_header].notna()].index.tolist()
        if not valid_rows: continue
        row_idx = random.choice(valid_rows)
        original_val = dirty_df.loc[row_idx, col_header]
        dirty_df.loc[row_idx, col_header] = violator

        # 4. Log
        ground_truth_records.append({
            'table': csv_fname,
            'row_idx': row_idx,
            'col_header': col_header,
            'original_val': original_val,
            'dirty_val': violator,
            'rule_type': rule_type,
            'target_rule': str(target_rule)
        })

    # Save clean and dirty to destination
    dst_folder = os.path.join(args.dst_dataset, table)
    if not os.path.exists(dst_folder): os.makedirs(dst_folder)

    input_df.to_csv(str(os.path.join(dst_folder, "clean.csv")), index=False)
    dirty_df.to_csv(str(os.path.join(dst_folder, "dirty.csv")), index=False)

if ground_truth_records:
    pd.DataFrame(ground_truth_records).to_csv(os.path.join(args.dst_dataset, "ground_truth_corruptions.csv"), index=False)
    print("Done. Saved clean.csv and dirty.csv to each destination folder and generated ground_truth_corruptions.csv")