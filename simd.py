from __future__ import annotations
import typing as t
import dataclasses

# cache wrapper removes type information from wrapped function:
if t.TYPE_CHECKING:

    def cache[T](x: T, /) -> T:
        ...

else:
    from functools import cache


@cache
def mask(n: int, /) -> int:
    """
    11...11
    ^^^^^^^ - n bits
    """
    return ~(~0 << n)


@cache
def mask_array(n: int, bi: int, /) -> int:
    """
    00...001 00...001 ... 00...001 00...001
           ^        ^            ^        ^ - n items
    ^^^^^^^^ ^^^^^^^^     ^^^^^^^^ ^^^^^^^^ - bi bits each
    """
    # return int(f'{1:0{bi}b}' * n, 2)
    return ~(~0 << n * bi) // ~(~0 << bi)


@cache
def _mask_array_val(n: int, bi: int, /) -> int:
    """
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
            ^          ^          ^          ^          ^
    """
    # works like array of 1
    return mask_array(n, bi)


@cache
def _mask_val(n: int, bp: int, bv: int, /) -> int:
    """
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
         ^^^^       ^^^^       ^^^^       ^^^^       ^^^^
    """
    # works like array of MAXVAL
    return _mask_array_val(n, bp + bv) * mask(bv)


@cache
def _mask_array_pad(n: int, bp: int, bv: int, /) -> int:
    """
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
       ^          ^          ^          ^          ^
    """
    # works like array of MAXVAL+1
    return mask_array(n, bp + bv) << bv


@cache
def _mask_pad(n: int, bp: int, bv: int, /) -> int:
    """
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
    ^^^^       ^^^^       ^^^^       ^^^^       ^^^^
    """
    return _mask_array_pad(n, bp, bv) * mask(bp)


@dataclasses.dataclass
class Shape:
    pass


class A:
    """
    A(0x0504030201, *(4, 4, 5)):

       0    5     0    4     0    3     0    2     0    1
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
    ^^^^       ^^^^       ^^^^       ^^^^       ^^^^      - padding
         ^^^^       ^^^^       ^^^^       ^^^^       ^^^^ - values
    ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^ - items
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ - data
            4          3          2          1          0 - items are indexed from least significant to most significant

    assumptions:
      padding bits are always filled with zeros
      all attrs makes sense
      .data is nonnegative
      if operation requires padding, then it is performed on arrays that do have enough padding, otherwise behaviour is unspecified

    """

    data: int
    bits_val: int
    bits_pad: int
    length: int

    def __init__(self, data: int, bv: int, bp: int, length: int, /) -> None:
        self.data = data
        self.bits_val = bv
        self.bits_pad = bp
        self.length = length

    def __eq__(self, other: object, /) -> bool:
        if other.__class__ is not A:
            return NotImplemented
        assert isinstance(other, A), other
        return self._shape == other._shape and self.data == other.data

    def __ne__(self, other: object, /) -> bool:
        if other.__class__ is not A:
            return NotImplemented
        assert isinstance(other, A), other
        return self._shape != other._shape or self.data != other.data

    def _from_const(self, n: int, /) -> A:
        return A(self._mask_array_val * n, *self._shape)

    @property
    def _shape(self, /) -> tuple[int, int, int]:
        return self.bits_val, self.bits_pad, self.length

    @property
    def _bits_item(self, /) -> int:
        return self.bits_pad + self.bits_val

    @property
    def mask(self, /) -> int:
        """
        0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
        ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^
        """
        return mask(self._bits_item * self.length)

    @property
    def _mask_array_val(self, /) -> int:
        return _mask_array_val(self.length, self._bits_item)

    @property
    def _mask_val(self, /) -> int:
        return _mask_val(self.length, self.bits_pad, self.bits_val)

    @property
    def _mask_array_pad(self, /) -> int:
        return _mask_array_pad(self.length, self.bits_pad, self.bits_val)

    @property
    def _mask_pad(self, /) -> int:
        return _mask_pad(self.length, self.bits_pad, self.bits_val)

    def _get_item(self, i: int, /) -> int:
        return self.data >> (i * self._bits_item) & mask(self._bits_item)

    def _get_padval(self, i: int, /) -> tuple[int, int]:
        item = self._get_item(i)
        pad = item >> self.bits_val
        val = item & mask(self.bits_val)
        return pad, val

    def _get_pad(self, i: int, /) -> int:
        return self._get_padval(i)[0]

    def _get_val(self, i: int, /) -> int:
        return self._get_padval(i)[1]

    def __iter__(self) -> t.Iterator[int]:
        # TODO: do it in O(n), not O(n^2)
        for i in range(self.length):
            yield self._get_val(i)

    def __str__(self, /) -> str:
        res = []
        for i in range(self.length):
            pad, val = self._get_padval(i)
            res += [f"{pad:0{self.bits_pad}b}_{val:0{self.bits_val}b}"]
        return f"[{", ".join(res)}]"

    def __repr__(self, /) -> str:
        bits_per_hex_digit = 4
        hex_digits = self._bits_item * self.length / bits_per_hex_digit
        hex_digits = hex_digits.__ceil__()
        return f"{self.__class__.__qualname__}({self.data:#0{hex_digits+2}x}, *{self._shape})"

    def __add__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        if self._shape != other._shape:
            raise ValueError
        n1 = self.data
        n2 = other.data
        # perform addition:
        n = n1 + n2
        # fill padding with zeros:
        n &= self._mask_val
        return A(n, *self._shape)

    def __sub__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        if self._shape != other._shape:
            raise ValueError
        n1 = self.data
        n2 = other.data
        # store 1 in each padding block:
        n1 |= self._mask_array_pad
        # perform subtraction:
        n = n1 - n2
        # fill padding with zeros:
        n &= self._mask_val
        return A(n, *self._shape)

    def __mul__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            # multiplication by constant
            n = self.data
            n *= other
            n &= self._mask_val
            return A(n, *self._shape)

        if self._shape != other._shape:
            raise ValueError

        n1 = self.data
        n2 = other.data
        n = 0
        for i in range(self.bits_val):
            b = n2
            b >>= i
            b &= self._mask_array_val
            b *= mask(self.bits_val)
            b &= n1
            b <<= i
            n += b

        n &= self._mask_val
        return A(n, *self._shape)

    def __and__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        if self._shape != other._shape:
            raise ValueError
        n1 = self.data
        n2 = other.data
        n = n1 & n2
        return A(n, *self._shape)

    def __or__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        if self._shape != other._shape:
            raise ValueError
        n1 = self.data
        n2 = other.data
        n = n1 | n2
        return A(n, *self._shape)

    def __xor__(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        if self._shape != other._shape:
            raise ValueError
        n1 = self.data
        n2 = other.data
        n = n1 ^ n2
        return A(n, *self._shape)

    def __rshift__(self, b: int, /) -> A:
        n = self.data
        n >>= b
        n &= self._mask_val
        return A(n, *self._shape)

    def __lshift__(self, b: int, /) -> A:
        n = self.data
        n <<= b
        n &= self._mask_val
        return A(n, *self._shape)

    def __invert__(self, /) -> A:
        n = self.data
        n ^= self._mask_val
        return A(n, *self._shape)

    def __neg__(self, /) -> A:
        n = self.data
        n = self._mask_array_pad - n  # assumes there is a padding?
        # n = self._mask_val + self._mask_array_val - n # works in the same way
        return A(n, *self._shape)

    def is_true(self, /) -> A:
        n = self.data
        # store 1s in each padding block:
        n |= self._mask_array_pad
        # subtract 1:
        n -= self._mask_array_val
        # isolate the first padding bit: (it will underflow to 0, iff the starting value was 0)
        n >>= self.bits_val
        n &= self._mask_array_val
        return A(n, *self._shape)

    def is_false(self, /) -> A:
        n = self.is_true().data
        n ^= self._mask_array_val
        return A(n, *self._shape)

    def eq(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        return (self ^ other).is_false()

    def ne(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        return (self ^ other).is_true()

    def lt(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        # (self-other)<0
        n = (self - other).data
        # move sign bit to the beginning
        n >>= self.bits_val - 1
        n &= self._mask_array_val
        return A(n, *self._shape)

    def gt(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        # it is basically (other-self)<0
        n = (other - self).data
        # move sign bit to the beginning
        n >>= self.bits_val - 1
        n &= self._mask_array_val
        return A(n, *self._shape)

    def le(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        return self.lt(other) | self.eq(other)

    def ge(self, other: A | int, /) -> A:
        if isinstance(other, int):
            other = self._from_const(other)
        return self.gt(other) | self.eq(other)
