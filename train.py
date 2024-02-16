import os
import torch
import json
from math import exp

import click
from sentencepiece import SentencePieceProcessor
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
from model import *
import wandb

_ROOT = os.path.abspath(os.path.dirname(__file__))
LOG_DIR = _ROOT + "/log"
MODEL_DIR = _ROOT + "/model"


_ROOT = os.path.abspath(os.path.dirname(__file__))
BPE_MODEL_PATH = _ROOT + "/model/tokenizer.model"
tokenizer = SentencePieceProcessor(model_file=BPE_MODEL_PATH)


def preprocess(data, max_length=512):
    inputs = "<s>" + tokenizer.encode_as_ids(data["text"])
    targets = tokenizer.encode_as_ids(data["text"]) + "</s>"

    inputs = [inputs + [tokenizer.pad_token_id] * (max_length - len(inputs))]
    targets = [targets + [tokenizer.pad_token_id] * (max_length - len(targets))]
    return inputs[:512], targets[:512]


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = preprocess(self.data[idx])
        return x, y


@click.command()
@click.option('--num-layers', type=int, default=8, show_default=True, help="No. of decoder layers")
@click.option('--embedding-size', type=int, default=768, show_default=True, help="Embedding size")
@click.option('--num-heads', type=int, default=8, show_default=True, help="Number of heads")
@click.option('--dff', type=int, default=3072, show_default=True, help="Filter Size")
@click.option('--max-seq-len', type=int, default=515, show_default=True, help="Seq length")
@click.option('--vocab-size', type=int, default=24512, show_default=True, help="Vocab size")
@click.option('--batch-size', type=int, default=64, show_default=True, help="optimizer type")
@click.option('--learning-rate', type=float, default=0.001, show_default=True, help="learning rate")
def train(num_layers, embedding_size, num_heads, dff, max_seq_len, vocab_size,
          batch_size, learning_rate):
    train_dataset = load_from_disk("openwebtext_train")
    test_dataset = load_from_disk("openwebtext_test")

    train_loader = DataLoader(CustomDataset(train_dataset), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(CustomDataset(test_dataset), batch_size=batch_size, shuffle=False)

    config = {}
    model = LLAMA(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = model.configure_optimizers()

    counter = 0
    # Training loop
    for epoch in range(5):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            perplexity = exp(loss)
            loss.backward()
            optimizer.step()
            print(f"Loss {loss}, Perplexity: {perplexity:.2f}")
            wandb.log({"Train Loss": loss, "Train Perplexity": perplexity}, step=counter)
            counter += 1

        # Evaluation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            for inputs, targets in test_loader:
                logits, loss = model(inputs, targets)
                total_loss += loss.item() * inputs.size(0)
                perplexity = exp(loss)
            total_samples += inputs.size(0)

            avg_loss = total_loss / total_samples
            print(f"Epoch {epoch + 1}, Avg. Loss: {avg_loss}")
            wandb.log({"Val Loss": total_loss, "Val Perplexity": perplexity}, step=epoch)


if __name__ == "__main__":
    train()
