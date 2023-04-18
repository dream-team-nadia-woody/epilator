from dataclasses import dataclass
from typing import Callable, Union, Self

import numpy as np
from numpy.typing import ArrayLike
from video.conversion import Converter
AGG_FUNCS = {
    'sum': lambda x: np.sum(x, axis=1,dtype=np.uint64),
    'mean': lambda x: np.mean(x, axis=1, dtype=np.float64),
    'absum': lambda x: np.abs(np.sum(x,axis=1,dtype=np.uint64))

}


AggregatorFunc = Union[Callable, str]


@dataclass
class Channel:
    '''a wrapper class representing
    a color channel of a `Video`
    '''
    channel_name: str
    channel: np.ndarray
    converter: Converter

    def agg(self, agg: AggregatorFunc) -> ArrayLike:
        inline = self.channel.reshape((self.channel.shape[0], -1))
        if isinstance(agg, Callable):
            return agg(inline)
        return AGG_FUNCS[agg](inline)

    def pct_change(self, n: int,
                   agg: AggregatorFunc = AGG_FUNCS['sum']) -> ArrayLike:
        agg_arr = self.agg(agg).astype(np.float64)
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            pct_change_arr = (agg_arr - shifted_arr) / shifted_arr
        pct_change_arr[np.isnan(pct_change_arr)] = 0
        return pct_change_arr * 100

    def difference(self, n: int,
                   agg: AggregatorFunc = AGG_FUNCS['mean']) -> ArrayLike:
        agg_arr = self.agg(agg)
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        return agg_arr - shifted_arr

    def mask(self, min_threshold: int = 190, max_threshold: int = 255) -> Self:
        min_threshold = np.uint8(min_threshold)
        max_threshold = np.uint8(max_threshold)
        channel = self.channel.copy()
        mask = np.logical_and(
            channel >= min_threshold,
            channel <= max_threshold)
        channel[~mask] = 0
        return Channel(self.channel_name, channel, self.converter)
