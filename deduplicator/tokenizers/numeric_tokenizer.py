#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Iterable

from deduplicator.tokenizers.base_tokenizer import BaseTokenizer


class NumericTokenizer(BaseTokenizer):
    """
    Enables numerical comparisons of integers or floating point numbers.

    Implementation of the idea of Vatsalan and Christen:
    - Privacy-preserving matching of similar patients,
    Journal of Biomedical Informatics, 2015)
    - https://github.com/data61/clkhash/blob/master/clkhash/comparators.py#L90


    The numerical distance between two numbers relate to the similarity of the tokens
    produces by this comparison class.
    The main idea is to encode a number's neighbourhood such that the neighbourhoods
    of close numbers overlap, e.g. the neighbourhood of
    x=21 is 19, 20, 21, 22, 23
    and the neighbourhood of
    y=23 is 21, 22, 23, 24, 25
    These two neighbourhoods share three elements.
    The overlap of the neighbourhoods of two numbers increases
    the closer the numbers are to each other.

    There are two parameters to control the overlap:
    - `threshold_distance`:
    the maximum distance which leads to an non-empty overlap.
    Neighbourhoods for points which are further apart have no elements in common. (*)

    - `resolution`:
    Controls how many tokens are generated. (the `b` in the paper).
    Given an interval of size `threshold_distance`
    it's created 'resolution tokens to either side of the mid-point plus one token for
    the mid-point.
    Thus, 2 * `resolution` + 1 tokens in total.
    A higher resolution differentiates better between different values,
    but should be chosen such that it plays nicely
    with the overall Bloom filter size and insertion strategy.

    (*) There are several tricks for handling more complex cases:
    Firstly have to quantize the inputs to multiples of `threshold_distance` /
    (2 * `resolution`), in order to get comparable neighbourhoods.

    e.g. if `threshold_distance` as 8 was chosen and a `resolution` as 2,
    then, without quantization, the neighbourhood of x=25
    would be [21, 23, 25, 27, 29] and
    for y=26 [22, 24, 26, 28, 30], resulting in no overlap.

    The quantization ensures that the inputs are mapped onto a common grid.
    In our example, the values would be
    quantized to even numbers (multiples of 8 / (2 * 2) = 2).
    Thus x=25 would be mapped to 26.
    The quantization has the side effect that sometimes two values which are further
    than `threshold_distance` but not more than `threshold_distance` + 1/2 quantization
    level apart can share a common token.
    For instance,
    a=24.99 would be mapped to 24 with a neighbourhood of  [20, 22, 24, 26, 28],
    and
    b=16 neighbourhood is [12, 14, 16, 18, 20].

    The output tokens based on the neighbourhood is produced in the following way:
    instead of creating a neighbourhood around the quantized input with values
    dist_interval = `threshold_distance` / (2 * `resolution`) apart,
    it's multiply all values by (2 * `resolution`) instead.

    This saves the division, which can introduce numerical inaccuracies.
    Thus, the tokens for x=25 are [88, 96, 104, 112, 120].

    Also there is a dealing with floating point numbers by quantizing them to integers
    by multiplying them with 10 ** `fractional_precision`
    and then rounding them to the nearest integer.
    Thus, we don't support to full range of floats, but the subset between:
    2.2250738585072014e-(308 - fractional_precision - log(resolution, 10))
    and
    1.7976931348623157e+(308 - fractional_precision - log(resolution, 10))
    """

    def __init__(
            self,
            threshold_distance: float = 1,
            resolution: int = 1,
            fractional_precision: int = 0,
    ) -> None:
        """
        :param threshold_distance: maximum detectable distance.
        Points that are further apart won't have tokens in common.
        :param resolution: controls the amount of generated tokens.
        Total number of tokens will be 2 * resolution + 1
        :param fractional_precision: number of digits after the point to be considered
        """
        # check that there is enough precision to have non-zero threshold_distance
        if not threshold_distance > 0:
            raise ValueError(
                'threhold_distance has to be positive, but was {}'.format(
                    threshold_distance
                )
            )
        if resolution < 1:
            raise ValueError(
                'resolution has to be greater than zero, but was {}'.format(resolution)
            )
        if fractional_precision < 0:
            raise ValueError(
                'fractional_precision cannot be less than zero, but was {}'.format(
                    fractional_precision
                )
            )
        # instead of dividing threshold distance as in the paper,
        # we rather multiply the inputs by 'resolution' and then
        # use threshold_distance as distance_interval
        # (saves a division which would need more precision)
        self.distance_interval = int(
            round(threshold_distance * pow(10, fractional_precision))
        )
        if self.distance_interval == 0:
            raise ValueError(
                'not enough fractional precision to encode threshold_distance'
            )
        self.resolution = resolution
        self.fractional_precision = fractional_precision

    def _tokenize(self, word: str) -> Iterable[str]:
        word = str(word).strip()
        if len(word) == 0:
            return tuple()
        try:
            v = int(word, base=10)  # we try int first, so we don't loose precision
            if self.fractional_precision > 0:
                v *= pow(10, self.fractional_precision)
        except ValueError:
            v_float = float(word)
            if self.fractional_precision > 0:
                v = int(round(v_float * pow(10, self.fractional_precision)))
            else:
                v = int(v_float)
        v = v * 2 * self.resolution
        residue = v % self.distance_interval

        if residue == 0:
            v = v
        elif residue < self.distance_interval / 2:
            v = v - residue
        else:
            v = v + (self.distance_interval - residue)

        return [
            str(v + i * self.distance_interval)
            for i in range(-self.resolution, self.resolution + 1)
        ]


if __name__ == '__main__':
    tokenizer = NumericTokenizer(
        threshold_distance=1, resolution=1, fractional_precision=0
    )
    print(tokenizer('25'))
    print(tokenizer('25.89'))
    print(tokenizer('26'))
