import os
import click
import sentencepiece as spm
from datasets import load_dataset


num_proc_load_dataset = 4
# Just downloading 5% of data for faster process
dataset = load_dataset("openwebtext", split="train[:5%]", num_proc=num_proc_load_dataset)

_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESS_DATA_TXT = _ROOT + "/data/processed.txt"
BPE_MODEL_PATH = _ROOT + "/model/tokenizer.model"
PROCESS_DATA_PATH = _ROOT + "/data/processed_data"


def process_text():
    print("Pre-processing the text data.....")
    with open(PROCESS_DATA_TXT, "w", encoding="utf-8") as file:
        # Write training data
        for row in dataset["train"]:
            file.write(row["text"] + "\n")
        # Write testing data
        for row in dataset["val"]:
            file.write(row["text"] + "\n")
    dataset.save_to_disk(PROCESS_DATA_PATH)


def train_byte_pair_encoding(vocab_size):
    print("Training BytePair encoding......")

    spmcmd = ('--input={spm_input} --model_prefix={spm_model} --vocab_size={vocab_size} '
              '--user_defined_symbols=<s>,</s> --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=1 '
              '--bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]').format(
        spm_input=PROCESS_DATA_PATH, spm_model=BPE_MODEL_PATH, vocab_size=vocab_size)
    spm.SentencePieceTrainer.train(spmcmd)


@click.command()
@click.option('--vocab-size', type=int, default=32000, show_default=True, help="vocab size")
def train(vocab_size):
    process_text()
    train_byte_pair_encoding(vocab_size)
    print("Pre-processing is done............")


if __name__ == "__main__":
    train()
