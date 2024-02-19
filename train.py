import os
import torch
import json
from math import exp

import click
from sentencepiece import SentencePieceProcessor
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from model import *
import wandb

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"

_ROOT = os.path.abspath(os.path.dirname(__file__))
BPE_MODEL_PATH = _ROOT + "/model/tokenizer.model"
PROCESS_DATA_TXT = _ROOT + "/data/processed.txt"
tokenizer = SentencePieceProcessor(model_file=BPE_MODEL_PATH)
print(tokenizer.pad_id())


def preprocess(doc, max_length=512):
    print(doc.strip())
    inputs = tokenizer.encode_as_ids("<s>" + doc)
    targets = tokenizer.encode_as_ids(doc + "</s>")

    print(inputs)
    print(tokenizer.encode_as_ids())

    #     inputs = [inputs + [tokenizer.pad_id()] * (max_length - len(inputs))]
    #     targets = [targets + [tokenizer.pad_id()] * (max_length - len(targets))]
    return inputs[:512], targets[:512]


class CustomDataset(Dataset):
    def __init__(self, data_file, sp_model):
        self.data_x = []
        self.data_y = []
        self.sp_model = sp_model

        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Tokenize the sentence and convert to IDs
                    tokens = [3] + self.sp_model.EncodeAsIds(line) + [4]
                    if len(tokens) < 32:
                        continue
                    # For language modeling task, x and y are the same sequence, shifted by one token
                    self.data_x.append(tokens[:-1])  # Input sequence (all tokens except the last one)
                    self.data_y.append(tokens[1:])  # Target sequence (all tokens except the first one)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return torch.tensor(self.data_x[idx], dtype=torch.long), torch.tensor(self.data_y[idx], dtype=torch.long)


def create_dataloader(dataset, batch_size):
    def collate_fn(batch):
        # Transpose batch of tuples
        batch_x, batch_y = zip(*batch)
        # Pad sequences in each batch
        batch_x = pad_sequence(batch_x, batch_first=True)
        batch_y = pad_sequence(batch_y, batch_first=True)
        return batch_x, batch_y

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloader


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=24512, show_default=True, help="Vocab size")
@click.option('--batch-size', type=int, default=8, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.0001, show_default=True, help="learning rate")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          batch_size, learning_rate):
    tokenizer = SentencePieceProcessor(model_file=_ROOT + "/data/tokenizer.model")
    dataset = CustomDataset(_ROOT + "/data/processed.txt", tokenizer)
    dataloader = create_dataloader(dataset, batch_size)

    config = {"vocab_size": 32000,
              "n_head": 12,
              "hidden_size": 768,
              "n_layer": 6,
              "n_embd": 768,
              "n_local_heads": 23,
              "n_local_kv_heads": 12,
              "eps": 1e-10,
              "max_len": 1024,
              "rope_theta": 1.0,
              "num_key_value_heads": 12,
              "attention_dropout": 0.25,
              "rms_norm_eps": 1.0,
              "weight_decay": 0.1,
              "betas": 0.00,
              "block_size": 1024}
    model = LLAMA(config)
    model._init_weights(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=config["weight_decay"])

    counter = 0
    # Training loop
    for epoch in range(1):
        wandb.init()
        # model.train()
        for batch_x, batch_y in dataloader:
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            logits, loss = model(batch_x.to(device), batch_y.to(device))
            perplexity = exp(loss)
            loss.backward()
            optimizer.step()
            print(f"Loss {loss}, Perplexity: {perplexity:.2f}")
            wandb.log({"Train Loss": loss, "Train Perplexity": perplexity}, step=counter)
            counter += 1

        # Evaluation
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for inputs, targets in dataloader:
                logits, loss = model(inputs.to(device), targets.to(device))
                total_loss += loss.item() * inputs.size(0)
                perplexity = exp(loss)
            total_samples += inputs.size(0)

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}, Avg. Loss: {avg_loss}")
            wandb.log({"Val Loss": total_loss, "Val Perplexity": perplexity}, step=epoch)


if __name__ == "__main__":
    train()
