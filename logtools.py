import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
import chardet
import pandas as pd
import faiss
from collections import Counter
import io
import sys

MISSING = 'FILE MISSING' # Output as log content when file is missing
EMPTY = 'EMPTY' # Output as log content when file is empty
READ_ERROR = 'READ ERROR' # Output as log content if error occurs while reading file

def read_log_file(log_file, preprocessor=None, max_entries=None, verbose=False):
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(log_file, 'rb') as f:
            encoding = chardet.detect(f.read())['encoding']
            if verbose:
                print(f"File {log_file}: encoding detected as {encoding}")
        with open(log_file, 'r', encoding=encoding) as f:
            try:
                lines = f.readlines()
            except UnicodeDecodeError:
                print(f"File {log_file}: tried encoding {encoding} but failed, returning [READ_ERROR]")
                return [READ_ERROR]
    # If lines is the empty list, return the EMPTY identifier
    if len(lines) == 0:
        return [EMPTY]
    if max_entries is not None and len(lines) > max_entries:
        half_lines = max_entries // 2
        lines = lines[:half_lines] + lines[-half_lines:]
    if preprocessor is not None:
        lines = preprocessor(lines)
    lines = [line.strip() for line in lines]
    # Drop first line if it is the empty string
    if len(lines) > 0 and lines[0] == '':
        lines = lines[1:]
    return lines
    
def get_entries_by_paths(df, log_data_dir, max_entries=None):
    # Returns a list of lists of entries.
    # If df has a column called 'log_entries', use that.
    if 'log_entries' in df.columns:
        print("Log entries are already in the dataframe, using them.")
        return df['log_entries'].tolist()
    # Otherwise, use the log_file column to get the paths to the log files.
    paths = log_data_dir + '/' + df['run'] + '/' + df['log_type'].astype('str')
    entries = [read_log_file(p, max_entries=max_entries) for p in paths]
    return entries

def trimmed_mean(xs, trim):
    # If trim is an integer, trim that many values from both ends.
    # If it is a float, trim that fraction of values from both ends.
    xs = np.array(xs)
    n = len(xs)
    if isinstance(trim, int):
        n_to_trim = trim
    elif isinstance(trim, float):
        n_to_trim = int(trim * n)
    else:
        raise ValueError("totrim must be an int or a float.")
    if n_to_trim == 0:
        return xs.mean()
    xs = np.sort(xs)
    return xs[n_to_trim:-n_to_trim].mean()
    
def get_extended_entries(entries, window_size):
    extended_entries = ["\n".join(entries[i:min(i+window_size,len(entries))]) for i in range(len(entries))]
    return extended_entries

def get_paths_by_log_type(log_type, files_df):
    # If log_type is a Path, convert it to a string
    if isinstance(log_type, os.PathLike):
        log_type = str(log_type)
    paths = files_df[files_df['log_type'].apply(str) == log_type]
    return paths

def get_mean_dists(D, I, df_ref_embeddings, k):
    """ 
    Get the mean distance to the k nearest embeddings in the reference set.
    Multiple occurrences of the same extended entry are counted separately.
    """
    # "count" is how many times the extended entry appears in the reference set.
    mean_dists = []
    for (dists, inds) in zip(D, I):
        counts = df_ref_embeddings.iloc[inds]['count'].values
        k_left = k
        i = 0
        sum_dist = 0.0
        while k_left > 0:
            to_take = min(k_left, counts[i])
            sum_dist += to_take * dists[i]
            k_left -= to_take
            i += 1
        mean_dist = sum_dist / k
        mean_dists += [mean_dist]
    return np.array(mean_dists)

def get_embedded_entry_df(log_type,
                          files_df,
                          log_data_source,
                          model,
                          window_size,
                          reference=True,
                          max_lines=None):
    """
    Produces the extended entries and their embeddings for a given log type.
    For the reference set, reference=True. For the holdout set, reference=False.
    For the reference set the counts of the extended entries are included.
    For the holdout set, the order of the extended entries is preserved.
    log_data_source is either the log data directory or a tuple (ids, entries).
    The latter is only allowed if reference=False.
    """
    print(f"Processing log {log_type}")

    # If log_data_source is a tuple, it contains indices and log entries.
    if isinstance(log_data_source, tuple):
        if reference:
            raise ValueError("log_data_source must be a directory if reference=True.")
        entry_indices, logs_as_entries = log_data_source
        logs_as_entries = [logs_as_entries]
    else:
        log_paths = get_paths_by_log_type(log_type, files_df)
        logs_as_entries = get_entries_by_paths(log_paths, log_data_source, max_entries=max_lines)
        entry_indices = None
    extended_entry_lists = [get_extended_entries(entries, window_size) for entries in logs_as_entries]
    
    if reference:
        c = Counter()
        for extended_entry_list in extended_entry_lists:
            c.update(extended_entry_list)
        df_lines = pd.DataFrame.from_dict(c, orient='index').reset_index()
        df_lines.columns = ['extended_entry', 'count']
    else:
        # In the test set, there is only one log per log type.
        # In addition to the extended entry, we also want to keep the original entry.
        df_lines = pd.DataFrame({'entry': logs_as_entries[0], 'extended_entry': extended_entry_lists[0]})
        if entry_indices is not None:
            df_lines['entry_id'] = entry_indices
    df_lines.insert(0, 'log_type', log_type)
    embeddings = model.encode(df_lines['extended_entry'].values)
    df_lines['embedding'] = list(embeddings)
    return df_lines

def partition_holdout_set(df_hold_dists, holdout_runs, test_run=None):
  if test_run is None:
      test_run = np.random.choice(holdout_runs)
      print(f"Test run: {test_run}")
  val_runs = np.array([r for r in holdout_runs if r != test_run])
  df_val_dists = df_hold_dists[df_hold_dists['run'].isin(val_runs)]
  df_test_dists = df_hold_dists[df_hold_dists['run'] == test_run]
  return df_val_dists, df_test_dists

def get_runs_files_df(log_data_dir, include=None):
    """
    Get a DataFrame with the runs and files in the log data directory.

    Parameters
    ----------
    log_data_dir : str
        The path to the directory containing the log data.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the runs and files in the log data directory
    """
    
    file_run_pairs = []
    
    # Iterate through the directories in the base directory
    for run in os.listdir(log_data_dir):
        run_path = os.path.join(log_data_dir, run)
        if os.path.isdir(run_path):
            for root, _, files in os.walk(run_path):
                for file in files:
                    relative_path = os.path.relpath(os.path.join(root, file), run_path)
                    file_run_pairs.append((relative_path, run))
    
    df = pd.DataFrame(file_run_pairs, columns=['log_type', 'run'])
    if include is not None:
        print(f"Include list provided. It contains {len(include)} log types. Filtering the DataFrame.")
        df = df.query("log_type in @include").reset_index(drop=True).copy()
    df.sort_values(by=['run', 'log_type'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def get_embedded_ref_entries_df(
    ref_df,
    log_data_dir,
    model,
    window_size,
    max_lines=None
):
    ref_log_types = sorted(ref_df['log_type'].unique().astype(str))
    entry_dfs = []
    for log_type in ref_log_types:
        df_entries = get_embedded_entry_df(
            log_type, ref_df, log_data_dir, model, window_size,
            reference=True, max_lines=max_lines)
        entry_dfs += [df_entries]
    ref_entries = pd.concat(entry_dfs)
    return ref_entries

def get_embedded_holdout_entries_df(
    ref_log_types,
    holdout_files_df,
    log_data_dir,
    model,
    window_size=4,
    max_lines=None):
    """
    Produces a dataframe with the extended entries and their embeddings
    for the holdout set.

    Parameters
    ref_log_types : list
        The list of log types to include in the holdout set.
    holdout_files_df : pd.DataFrame
        The DataFrame containing the paths of all files in the holdout set.
    log_data_dir : str
        The path to the directory containing the log data.
    model : SentenceTransformer
        The model to use for embedding the logs.
    window_size : int
        The size of the sliding window for creating extended entries.
    max_lines : int
        The maximum number of lines (entries) to read from each log file.

    Returns
    pd.DataFrame
        A DataFrame with the extended entries and their embeddings for the holdout set.
    """
    # This is unoptimized in the sense that it re-embeds
    # extended entries that have already been embedded in previous runs.

    holdout_runs = holdout_files_df['run'].unique()
    entry_dfs = []

    for run in holdout_runs:
        print(f"Getting holdout entries for run {run}")
        run_files_df = holdout_files_df[holdout_files_df['run'] == run]
        run_log_types = run_files_df['log_type'].unique().astype(str)
        valid_log_types = [l for l in run_log_types if l in ref_log_types]
        for log_type in valid_log_types:
            entries_df = get_embedded_entry_df(
                log_type, run_files_df, log_data_dir, model, window_size,
                reference=False, max_lines=max_lines)
            entries_df['run'] = run
            entry_dfs += [entries_df]
    embedded_holdout_entries = pd.concat(entry_dfs)
    return embedded_holdout_entries

def reference_holdout_split(all_files_df, holdout_share=0.1, test_runs=None):
    """
    Splits the DataFrame into reference and holdout sets based on the specified holdout share.
    If test_run is provided, it is guaranteed to be in the holdout set.

    Parameters:
    all_files_df (pd.DataFrame): The DataFrame containing the paths of all files in the dataset.
    holdout_share (float): The proportion of unique runs to include in the holdout set (default is 0.1).
    
    Returns:
    tuple: A tuple containing two DataFrames:
        - reference_files_df (pd.DataFrame): The DataFrame containing the reference runs.
        - holdout_files_df (pd.DataFrame): The DataFrame containing the holdout runs.
    """
    np.random.seed(0)
    runs = all_files_df['run'].unique()
    n_holdout_runs = int(len(runs)*holdout_share)
    print(f"Number of unique runs: {len(runs)}")
    print(f"Number of holdout runs: {n_holdout_runs}")
    np.random.shuffle(runs)
    if test_runs is not None and test_runs not in runs:
        raise ValueError(f"Test run {test_runs} not found in the DataFrame.")
    if test_runs is not None:
        # Move the test runs to the front of the list.
        non_test_runs = [r for r in runs if r != test_runs]
        runs = np.concatenate([test_runs, non_test_runs])
        # If test_runs is provided, n_holdout_runs must be at least len(test_runs).
        n_holdout_runs = max(n_holdout_runs, len(test_runs))
    holdout_runs = runs[:n_holdout_runs]
    reference_runs = runs[n_holdout_runs:]
    holdout_files_df = all_files_df[all_files_df['run'].isin(holdout_runs)].copy()
    reference_files_df = all_files_df[all_files_df['run'].isin(reference_runs)].copy()
    return reference_files_df, holdout_files_df

def calculate_distances(
    log_type,
    inference_entries_df,
    ref_embeddings_df,
    gpu_res,
    k=4,
    keep_highest_only=True,
    metric='squared_euclidean'):
    """
    Calculates the distances for all extended entries in log type 'log'
    over the inference set, which is a collection of extended entries
    and their embeddings. The inference set may contain multiple runs,
    in which case all runs are kept.
    """
    if metric not in ['squared_euclidean', 'euclidean', 'inner_product']:
        raise ValueError("Invalid metric. Must be 'squared_euclidean', 'euclidean' or 'inner_product'.")

    inference_entries = inference_entries_df.query("log_type == @log_type").copy()
    inference_entries_unique = inference_entries.drop_duplicates(subset=['extended_entry']).copy()

    ref_entries = ref_embeddings_df.query("log_type == @log_type").copy()

    d = len(inference_entries['embedding'].values[0])
    if metric == 'squared_euclidean':
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(np.stack(ref_entries['embedding'].values))
        D, I = index.search(np.stack(inference_entries_unique['embedding'].values), k)
    elif metric == 'euclidean':
        index = faiss.IndexFlatL2(d)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(np.stack(ref_entries['embedding'].values))
        D, I = index.search(np.stack(inference_entries_unique['embedding'].values), k)
        D = np.sqrt(D)
    elif metric == 'inner_product':
        index = faiss.IndexFlatIP(d)
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(np.stack(ref_entries['embedding'].values))
        D, I = index.search(np.stack(inference_entries_unique['embedding'].values), k)
        # Convert similarities to distances
        D = 1 - D
    mean_dists = get_mean_dists(D, I, ref_entries, k)
    inference_entries_unique.loc[:,'dist'] = mean_dists
    inference_entries_unique = inference_entries_unique[['extended_entry', 'dist']]
    ie_merged = inference_entries.merge(inference_entries_unique, on='extended_entry', how='inner')
    if not keep_highest_only:
        return ie_merged
    ie_highest = ie_merged.sort_values(by='dist', ascending=False).drop_duplicates(subset='run')
    return ie_highest

def calculate_distances_all_log_types(
        inference_entries_df,
        ref_embeddings_df,
        gpu_res,
        k=4,
        keep_highest_only=True,
        metric='squared_euclidean',
        ):
    """
    Calculates the distances for all log types in the inference set. Distances for each run
    are maintained separately as we do not yet know which run will be the test run.
    """

    # inference_entries_df must have an 'entry_id' column. It must not
    # contain duplicates.

    if not 'entry_id' in inference_entries_df.columns:
        raise ValueError("inference_entries_df must have an 'entry_id' column.")
    
    if inference_entries_df.duplicated(subset='entry_id').any():
        raise ValueError("inference_entries_df must not have duplicates in the entry_id column.")

    inference_log_types = inference_entries_df['log_type'].unique()
    dfs = []
    df = inference_entries_df.copy()

    for log in inference_log_types:
        dists = calculate_distances(log, df, ref_embeddings_df, gpu_res, k, keep_highest_only, metric)
        dfs += [dists]
    dfs = pd.concat(dfs)

    # The returned df must be in the same order as the input inference_entries_df.
    dfs = dfs.sort_values(by='entry_id').reset_index(drop=True)

    return dfs

def calculate_baselines(test_log_types, df_val_dists, test_run):
    baselines = []
    val_log_types = sorted(df_val_dists['log_type'].unique())
    for log_type in test_log_types:
        if log_type not in val_log_types:
            print(f"Skipping {log_type} as it is not in val_log_types.")
            baselines.append(np.nan)
            continue
        df_test_dists = df_val_dists.query("log_type == @log_type and run != @test_run").copy()
        # Keep the largest distance for each run (in the current log type).
        df_test_dists = df_test_dists.sort_values(by='dist', ascending=False).drop_duplicates(subset='run')
        trimmed_mean_dist = trimmed_mean(df_test_dists['dist'].values, 0.1)
        baselines.append(trimmed_mean_dist)
    baselines = np.array(baselines)
    baseline_df = pd.DataFrame(test_log_types, columns=['log_type'])
    baseline_df['baseline'] = baselines
    return baseline_df

def calculate_test_anomaly_scores(
    test_run_df,
    ref_entry_embeddings,
    k,
    res,
    baselines,
    metric='squared_euclidean'):
    """
    Given a dataframe of the test run extended entries and their embeddings,
    calculate the anomaly scores for each extended entry.
    """

    # If test_run_df lacks the entry_id column, make a copy and add it.
    # Otherwise, use the existing entry_id column.
    if 'entry_id' not in test_run_df.columns:
        test_run_df = test_run_df.copy()
        test_run_df.insert(0, 'entry_id', range(len(test_run_df)))
    
    distances_df = calculate_distances_all_log_types(
        test_run_df,
        ref_entry_embeddings,
        res,
        k,
        keep_highest_only=False,
        metric=metric,
        )
    
    # Subtract the baselines from the distances. Do not sort the distances.
    # baselines is a df with cols ['log_type', 'baseline']
    distances_df = distances_df.merge(baselines, on='log_type', how='left')
    distances_df['anomaly_score'] = distances_df['dist'] - distances_df['baseline']

    return distances_df

