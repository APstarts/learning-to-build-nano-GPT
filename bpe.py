import pickle


class BytePairTokenizer:
    def __init__(self, vocab_size):
        """Initialize the tokenizer.
        Args:
            vocab_size (int): Desired size of the vocabulary. The final number of tokens after training.
        """
        self.vocab_size = vocab_size
        self.merges = {}  # (int, int) --> int
        self.vocab = {}  # int --> bytes

    def get_stats(self, ids):
        """Takes in the list of raw byte ids to get the pairs with their counts representing the number of times the pairs occurred."""
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids, pair, idx):
        """This method is used to perform the merging of the intended pairs."""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, text):
        """Takes in the raw text that the user wants to train their tokenizer on."""
        # Step 1: convert to byte ids
        tokens = list(text.encode("utf-8"))
        ids = list(tokens)

        # Step 2: compute merges
        num_merges = self.vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i

            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        # step 3: build the vocab
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self.get_stats(tokens)

            pair = min(stats, key=lambda p: stats.get(p, float("inf")))

            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)  # gives out the new list of byte ids
        return tokens

    def decode(self, ids):
        """The list of byte ids. Typically, the output of the GPT kind of model you wish to decode into actual text."""
        byte_seq = b"".join(self.vocab[idx] for idx in ids)
        return byte_seq.decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.merges, self.vocab), f)

    def load(self, path):
        with open(path, "rb") as f:
            self.merges, self.vocab = pickle.load(f)
