"""Fixed character vocabulary; future variable letters are representable but untrained."""
import string
PAD,BOS,SEP,EOS=0,1,2,3
CHARS=sorted(set(string.ascii_lowercase+string.digits+' +-*=()%,;?|:'+ 'суммалогиакд'))
VOCAB=['<pad>','<bos>','<sep>','<eos>']+CHARS
IDS={c:i+4 for i,c in enumerate(CHARS)}


def encode(text):
    try:return [IDS[c] for c in text]
    except KeyError as error:raise ValueError(f'Unsupported character: {error.args[0]!r}') from error


def decode(ids):
    return ''.join(VOCAB[i] for i in ids if i>=4)


def prompt_ids(prompt):return [BOS]+encode(prompt)+[SEP]
