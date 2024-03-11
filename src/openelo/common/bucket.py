from .ordering import Ordering


def bucket(a: float, width: float) -> int:
    '''
    Given a number and the width of the bucket, return
    a rounded integer resembling a bucket.

        Args:
            a (:obj:`float`): Input number.

            width (:obj:`float`): Width of the bucket.

        Returns:
            :obj:`int`
    '''
    return int(round(a / width))


def same_bucket(a: float, b: float, width: float) -> bool:
    '''
    Given two numbers and a bucket width, return a boolean 
    if the two numbers are in the same bucket.

        Args:
            a (:obj:`float`): First input number.

            b (:obj:`float`): Second input number.

            width (:obj:`float`): Width of the bucket.

        Returns:
            :obj:`bool`

    '''
    return bucket(a, width) == bucket(b, width)


def cmp_by_bucket(a: float, b: float, width: float) -> Ordering:
    '''
    Given two numbers and a bucket width, return an ordering
    object comparing `a` and `b` to be `LESS`, `EQUAL`, or
    `GREATER`.

        Args:
            a (:obj:`float`): First input number.

            b (:obj:`float`): Second input number.

            width (:obj:`float`): Width of the bucket.

        Returns:
            :obj:`Ordering`: Return an ordering according to
            `a` and `b`.

    '''
    return Ordering.cmp(bucket(a, width), bucket(b, width))
