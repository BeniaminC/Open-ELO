from enum import IntEnum
from typing import Any, Self


class Ordering(IntEnum):
    LESS = -1
    EQUAL = 0
    GREATER = 1

    @classmethod
    def cmp(cls, cmp1: Any, cmp2: Any) -> Self:
        return cls((cmp1 > cmp2) - (cmp1 < cmp2))
