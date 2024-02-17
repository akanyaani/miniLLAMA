import os
import click
import sentencepiece as spm
from datasets import load_dataset

num_proc_load_dataset = 4
# Just downloading 5% of data for faster process
dataset = load_dataset("openwebtext", streaming=True)

## Downloading Sample Data from openwebtext, Can be used large corpus as well for my be language and custome models
sample_data = []
for i, row in enumerate(dataset["train"]):
    sample_data.append(row["text"])
    if i > 5000:
        break

_ROOT = os.path.abspath(os.path.dirname(__file__))
PROCESS_DATA_TXT = _ROOT + "/data/processed.txt"
BPE_MODEL_PATH = _ROOT + "/model/tokenizer.model"


def process_text():
    print("Pre-processing the text data.....")
    folder_path = os.path.dirname(PROCESS_DATA_TXT)  # Extract the parent directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(PROCESS_DATA_TXT, "w", encoding="utf-8") as file:
        # Write training data
        for text in sample_data:
            file.write(text + "\n")


def train_byte_pair_encoding(vocab_size):
    print("Training BytePair encoding......")
    folder_path = os.path.dirname(BPE_MODEL_PATH)  # Extract the parent directory
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    spmcmd = ('--input={spm_input} --model_prefix={spm_model} --vocab_size={vocab_size} '
              '--user_defined_symbols=<s>,</s> --hard_vocab_limit=false --model_type=bpe --pad_id=0 --unk_id=1 '
              '--bos_id=-1 --eos_id=-1 --pad_piece=[PAD] --unk_piece=[UNK]').format(
        spm_input=PROCESS_DATA_TXT, spm_model=BPE_MODEL_PATH, vocab_size=vocab_size)
    spm.SentencePieceTrainer.train(spmcmd)


@click.command()
@click.option('--vocab-size', type=int, default=32000, show_default=True, help="vocab size")
def train(vocab_size):
    process_text()
    train_byte_pair_encoding(vocab_size)
    print("Pre-processing is done............")


if __name__ == "__main__":
    train()
