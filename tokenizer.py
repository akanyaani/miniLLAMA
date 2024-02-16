from sentencepiece import SentencePieceProcessor


class Tokenizer():
    def __init__(self, model_path: str):
        self.sp_model = SentencePieceProcessor(model_file=model_path)

    def encode(self, seq):
        return self.sp_model.encode(seq)

    def decode(self, ids):
        return self.sp_model.decode(ids)
