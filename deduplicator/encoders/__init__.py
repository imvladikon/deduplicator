#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from deduplicator.encoders.phonetic_encoders import PhoneticEncoder, ConsonantEncoder
from deduplicator.encoders.date_encoder import DateEncoder
from deduplicator.encoders.geo_encoder import GeoEncoder
from deduplicator.encoders.integer_encoders import (FirstInteger,
                                                    LastInteger,
                                                    LargestInteger,
                                                    RoundInteger,
                                                    SortedIntegers)
from deduplicator.encoders.string_encoders import (FirstNChars,
                                                   LastNChars,
                                                   FirstNWords,
                                                   StringEncoder,
                                                   NLetterAbbreviation,
                                                   FirstNCharsLastWord)
