import csv
import datetime
import glob
import os
from collections import Counter

import click
import numpy as np
import sentencepiece as spm
import tqdm
from ftfy import fix_text

_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESS_DATA_PATH = _ROOT + "/data/processed.txt"
BPE_TSV_PATH = _ROOT + "/data/bpe_spm.tsv"
BPE_MODEL_PATH = _ROOT + "/model/tokenizer"

os.makedirs(_ROOT+"/model", exist_ok=True)
BOS_ID = 3
EOS_ID = 4


def process_text(text_files):
    print("Pre-processing the text data.....")
    file_writer = open(PROCESS_DATA_PATH, "w", encoding="utf-8")
    for file_name in tqdm.tqdm(text_files):
        fr = open(file_name, 'r',encoding="utf-8")
        file_writer.writelines([fix_text(line, normalization='NFKC') for line in fr.readlines()])
        fr.close
    file_writer.close()


def train_byte_pair_encoding(vocab_size):
    print("Training BytePair encoding......")
    token_dict = Counter()
    with open(PROCESS_DATA_PATH, 'r', encoding="utf-8") as fr:
        for line in tqdm.tqdm(fr):
            token_dict.update(line.lower().split())

    with open(BPE_TSV_PATH, 'w', newline='', encoding="utf-8") as f_output:
        tsv_output = csv.writer(f_output, delimiter='\t')
        for word in token_dict:
            tsv_output.writerow([word, token_dict[word]])

    spmcmd = ('--input={spm_input} --model_prefix={spm_model} --input_format=tsv --vocab_size={vocab_size} '
              '--user_defined_symbols=[SEP],[BOS],[EOS] --hard_vocab_limit=false --model_type=bpe --pad_id=0 '
              '--unk_id=1 --bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]').format(
        spm_input=BPE_TSV_PATH, spm_model=BPE_MODEL_PATH, vocab_size=vocab_size)
    spm.SentencePieceTrainer.train(spmcmd)




@click.command()
@click.option('--data-dir', type=str, default="./data/sample", show_default=True, help="training data path")
@click.option('--vocab-size', type=int, default=24512, show_default=True, help="byte pair vocab size")
@click.option('--min-seq-len', type=int, default=15, show_default=True, help="minimum sequence length")
@click.option('--max-seq-len', type=int, default=512, show_default=True, help="minimum sequence length")
def train(data_dir, vocab_size, min_seq_len, max_seq_len):
    text_files = glob.glob((data_dir + "/*.txt"))
    process_text(text_files)
    train_byte_pair_encoding(vocab_size)
    print("Pre-processing is done............")


if __name__ == "__main__":
	train()