from mmr_systems.common.ordering import Ordering


def bucket(a: float, width: float) -> int:
    return int(round(a / width))


def same_bucket(a: float, b: float, width: float) -> bool:
    return bucket(a, width) == bucket(b, width)


def cmp_by_bucket(a: float, b: float, width: float) -> Ordering:
    return Ordering.cmp(bucket(a, width), bucket(b, width))
