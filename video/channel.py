from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
from numpy.typing import ArrayLike
from video.conversion import Converter

AGG_FUNCS = {
    'sum': lambda x: np.sum(x, axis=1),
    'mean': lambda x: np.mean(x, axis=1)

}


AggregatorFunc = Union[Callable, str]


@dataclass
class Channel:
    '''a wrapper class representing
    a color channel of a `Video`
    '''
    channel: np.ndarray
    conversion: Converter

    def agg(self, agg: AggregatorFunc) -> ArrayLike:
        inline = self.channel.reshape((self.channel.shape[0], -1))
        if isinstance(agg, Callable):
            return agg(inline)
        return AGG_FUNCS[agg](inline)

    def pct_change(self, n: int,
                   agg: AggregatorFunc = AGG_FUNCS['mean']) -> ArrayLike:
        agg_arr = self.agg(agg)
        shifted_arr = np.roll(agg_arr, n)
        if np.issubdtype(shifted_arr.dtype, np.floating):
            shifted_arr[:n] = np.nan
        else:
            shifted_arr[:n] = 0
        return (agg_arr - shifted_arr) / shifted_arr
    def difference(self, n: int,
                agg: AggregatorFunc = AGG_FUNCS['mean']) -> ArrayLike:
        agg_arr = self.agg(agg)
        shifted_arr = np.roll(agg_arr, n)
        shifted_arr[:n] = np.nan
        return agg_arr - shifted_arr