#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple, Union

from deduplicator.encoders.base_encoder import BaseEncoder
from functools import lru_cache

from deduplicator.utils import check_backend_availability

try:
    from geopy.geocoders import Nominatim
except ImportError:
    pass

# the distance between geohashes based on matching characters, in meters.
_PRECISION = {
    0: 20000000,
    1: 5003530,
    2: 625441,
    3: 123264,
    4: 19545,
    5: 3803,
    6: 610,
    7: 118,
    8: 19,
    9: 3.71,
    10: 0.6,
}
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'


def geohash_encode(latitude: float, longitude: float, precision: int = 5) -> str:
    """
    Encode a position given in float arguments latitude, longitude to
    a geohash which will have the character count precision.
    """
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True
    while len(geohash) < precision:
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)
        even = not even
        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0
    return ''.join(geohash)


class ReverseGeoEncoder(BaseEncoder):

    def __init__(self, return_type='geohash', *args, **kwargs):
        check_backend_availability(self, 'geopy')
        super().__init__(*args, **kwargs)
        self.geolocator = Nominatim(user_agent="Mozilla/5.0 (X11; Linux i686) "
                                               "AppleWebKit/5341 (KHTML, like Gecko) "
                                               "Chrome/38.0.878.0 Mobile Safari/5341")

        assert return_type in ['lonlat', 'geohash']
        self.return_type = return_type

    @lru_cache(maxsize=1000)
    def _encode(self, address_query: str) -> Union[str, Tuple[float, float]]:
        location = self.geolocator.geocode(address_query)
        if self.return_type == 'lonlat':
            return (location.longitude, location.latitude)
        else:
            return geohash_encode(latitude=location.latitude,
                                  longitude=location.longitude)


class GeoEncoder(BaseEncoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = kwargs.get('precision', 5)

    def _encode(self, point: Tuple[float, float]) -> str:
        longitude, latitude = point
        return geohash_encode(latitude=latitude,
                              longitude=longitude,
                              precision=self.precision)
