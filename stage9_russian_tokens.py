"""Train-built Russian word inputs; original output alphabet stays unchanged.

Unknown words fall back to Cyrillic characters. No word is mapped to an action,
variable, numerical value or answer by this tokenizer.
"""
import re
from stage6_tokens import VOCAB, IDS, BOS, SEP, EOS, PAD, encode as encode_output, decode

RUSSIAN = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
WORDS = re.compile(r'[а-яё]+')


def vocabulary(prompts):
    words = sorted({m.group() for text in prompts for m in WORDS.finditer(text.lower())})
    characters = sorted(set(RUSSIAN + '.!«»—\n\t') - set(IDS))
    return dict(tokens=list(VOCAB) + characters + ['word:' + w for w in words],
        output_vocabulary=list(VOCAB), words=words, extra_characters=characters)


class Tokenizer:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        assert vocabulary['output_vocabulary'] == list(VOCAB)
        self.ids = {value: i for i, value in enumerate(vocabulary['tokens'])}
        self.word_ids = {word: self.ids['word:' + word] for word in vocabulary['words']}

    def encode(self, text):
        text = text.lower(); result = []; cursor = 0
        def characters(value):
            for char in value:
                if char not in self.ids: raise ValueError(f'Unsupported input character: {char!r}')
                result.append(self.ids[char])
        for match in WORDS.finditer(text):
            characters(text[cursor:match.start()])
            if match.group() in self.word_ids: result.append(self.word_ids[match.group()])
            else: characters(match.group())
            cursor = match.end()
        characters(text[cursor:])
        return result

    def prompt(self, text): return [BOS] + self.encode(text) + [SEP]

    def example(self, text, target):
        prefix = self.prompt(text)
        return prefix + encode_output(target) + [EOS], len(prefix)
