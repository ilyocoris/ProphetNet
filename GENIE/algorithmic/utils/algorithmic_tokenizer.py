from transformers import PreTrainedTokenizer

class AlgorithmicTokenizer(PreTrainedTokenizer):
    def __init__(
            self, 
            vocab = ["[SEP]", "[CLS]", "[UNK]", "[PAD]", "+", "-", "*", "p", "q", "r", "s", "t", "m"] + [str(n) for n in list(range(0,113))], 
            *args, 
            **kwargs
        ):
        super().__init__(*args, **kwargs)

        
        self.vocab = {word.strip(): idx for idx, word in enumerate(vocab)}

    def _tokenize(self, text):
        # Implement your custom tokenization logic here
        tokens = text.split()  # Replace this with your own tokenization logic
        return tokens

    def _convert_token_to_id(self, token):
        # Convert a token to its corresponding ID in the vocabulary
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        # Convert an ID to its corresponding token in the vocabulary
        return list(self.vocab.keys())[index]

def get_algorithmic_tokenizer(vocab = ["[SEP]", "[CLS]", "[UNK]", "[PAD]", "+", "-", "*", "p", "q", "r", "s", "t", "m"] + [str(n) for n in list(range(0,113))]):
    return AlgorithmicTokenizer(vocab=vocab, unk_token="[UNK]", pad_token="[PAD]")
    