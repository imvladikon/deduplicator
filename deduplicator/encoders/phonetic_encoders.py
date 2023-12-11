#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import groupby
from typing import Optional, Callable

from abydos.fingerprint import Phonetic as PhoneticFingerprint
from abydos.phonetic import DoubleMetaphone, Soundex, Metaphone, NYSIIS, BeiderMorse, \
    SpanishMetaphone
from deduplicator.tokenizers import WhitespaceSpaceTokenizer

from deduplicator.encoders.base_encoder import BaseEncoder


class PhoneticEncoder(BaseEncoder):

    def __init__(self,
                 algorithm: str = "double_metaphone",
                 max_token_length: int = 2,
                 max_length: int = 4,
                 tokenizer: Optional[Callable] = None,
                 *args,
                 **kwargs):
        """
        Phonetic encoder or extractor for strings based on abydos library.
        Args:
            algorithm: phonetic algorithm to use.
            Options: soundex, metaphone, nysiis, beidermorse, spanish_metaphone
            max_token_length: some phonetic algorithms could produce
            several encoded tokens for a single word.
            this parameter limits the number of tokens to be produced.
            max_length: maximum length of the encoded string.
            tokenizer:  tokenizer function to be used.
            If None, the default tokenizer is used.
            *args:
            **kwargs:
        """
        super(PhoneticEncoder, self).__init__(*args, **kwargs)
        if algorithm == "soundex":
            phonetic_algorithm = Soundex()
        elif algorithm == "metaphone":
            phonetic_algorithm = Metaphone()
        elif algorithm == "nysiis":
            phonetic_algorithm = NYSIIS()
        elif algorithm == "beidermorse":
            phonetic_algorithm = BeiderMorse()
        elif algorithm == "spanish_metaphone":
            phonetic_algorithm = SpanishMetaphone()
        elif algorithm == "double_metaphone":
            phonetic_algorithm = DoubleMetaphone()
        else:
            raise ValueError(
                f"Unknown phonetic algorithm: {algorithm}"
                "to additional algorithms, please refer to abydos library "
                "or specific libraries like https://github.com/roddar92/russian_soundex"
                "(Russian, English, Sweden, Estonian, Finnish, etc. phonetic)"
            )
        self.encoder = PhoneticFingerprint(phonetic_algorithm=phonetic_algorithm)
        self.max_length = max_length
        self.max_token_length = max_token_length
        self.tokenizer = tokenizer or WhitespaceSpaceTokenizer()

    def _encode(self, value: str) -> str:
        if not value:
            return ""
        values = self.tokenizer(value.lower())
        values = sorted(
            self.encoder.fingerprint(v)[: self.max_token_length] for v in values
        )
        values = "".join(values)
        return values.strip()[: self.max_length]


class ConsonantEncoder(BaseEncoder):

    def __init__(self, variant=1, doubles=True, vowels=None, *args, **kwargs):
        """Initialize Consonant instance.

        Parameters
        ----------
        variant : int
            Selects between Taft's 3 variants, which assign to the vowel set
            one of:

                1. A, E, I, O, & U
                2. A, E, I, O, U, W, & Y
                3. A, E, I, O, U, W, H, & Y

        doubles : bool
            If set to False, multiple consonants in a row are conflated to a
            single instance.
        vowels : list, set, or str
            Setting vowels to a non-None value overrides the variant setting
            and defines the set of letters to be removed from the input.
        """
        super().__init__(*args, **kwargs)
        self._vowels = vowels
        self._doubles = doubles

        if self._vowels is None:
            self._vowels = set('AEIOU')
            if variant > 1:
                self._vowels.add('W')
                self._vowels.add('Y')
            if variant > 2:
                self._vowels.add('H')
        else:
            self._vowels = {_.upper() for _ in self._vowels}

    def _encode(self, word):
        word = word.upper()
        # remove repeats if in -D variant
        if not self._doubles:
            word = ''.join(char for char, _ in groupby(word))
        # remove vowels
        word = word[:1] + ''.join(_ for _ in word[1:] if _ not in self._vowels)
        return word
