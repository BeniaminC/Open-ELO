from enum import IntEnum
from typing import Any, Self


class Ordering(IntEnum):
    '''
    An integer enumerator between -1 and 1 for ordering. Values
    are `LESS` (-1), `EQUAL` (0), and `GREATER` (1).
    '''
    LESS = -1
    EQUAL = 0
    GREATER = 1

    @classmethod
    def cmp(cls, cmp1: Any, cmp2: Any) -> Self:
        '''
        A class method to return an `Ordering` object based
        on two comparable objects.
        '''
        return cls((cmp1 > cmp2) - (cmp1 < cmp2))
