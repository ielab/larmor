import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, T5ForConditionalGeneration
import torch
from tqdm import tqdm

import glob
from ranx import Qrels, Run
from ranx import fuse

#

def fuse_runs(run_file_dir):
    run_files = glob.glob(run_file_dir + "/*.txt")
    runs = []
    for run_file in tqdm(run_files, desc='loading run files'):
        runs.append(Run.from_file(run_file, kind="trec"))
    fusion_run = fuse(
        runs=runs,
        method="rrf")
    return fusion_run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_file_dir", type=str)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()

    run_files = glob.glob(args.run_file_dir)
    print(run_files)
    runs = []
    for run_file in tqdm(run_files, desc='loading run files'):
        runs.append(Run.from_file(run_file, kind="trec"))


    print('fusing runs')
    fusion_run = fuse(
        runs=runs,
        method="rrf",
    )

    if args.fusion_save_path is not None:
        fusion_run.save(args.save_path, kind='trec')
