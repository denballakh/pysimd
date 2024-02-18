from __future__ import annotations
import typing as t
import dataclasses

T = t.TypeVar('T')

# cache wrapper removes type information from wrapped function:
if t.TYPE_CHECKING:

    def cache(x: T, /) -> T: ...

else:
    from functools import cache


@cache
def mask(n: int, /) -> int:
    """```py
    11...11
    ^^^^^^^ - n bits
    ```"""
    return ~(~0 << n)


@cache
def mask_array(n: int, bi: int, /) -> int:
    """```py
    00...001 00...001 ... 00...001 00...001
           ^        ^            ^        ^ - n items
    ^^^^^^^^ ^^^^^^^^     ^^^^^^^^ ^^^^^^^^ - bi bits each
    ```"""
    # TODO: is this the fastest thing? can this be made faster for bi==8 case?
    return ~(~0 << n * bi) // ~(~0 << bi)


@dataclasses.dataclass(frozen=True)
class S:
    # TODO: do we need these caches?
    """
    shape of an array
    """
    _: dataclasses.KW_ONLY
    len: int  # number of items
    bv: int  # number of value bits in each item
    bp: int  # number of padding bits in each item

    def __post_init__(self, /) -> None:
        assert self.len >= 0, f'length should be non-negative: {self}'
        assert self.bv > 0, f'value width should be positive: {self}'
        assert self.bp > 0, f'padding width should be positive: {self}'

    @property
    def bi(self, /) -> int:
        return self.bp + self.bv

    @property
    @cache
    def mask(self, /) -> int:
        """```py
        0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
        ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^  ^^^^ ^^^^
        ```"""
        return mask(self.bi * self.len)

    @property
    @cache
    def mask_array_val(self, /) -> int:
        """```py
        0000 0001  0000 0001  0000 0001  0000 0001  0000 0001
                ^          ^          ^          ^          ^
        ```"""
        # works like array of 1
        return mask_array(self.len, self.bi)

    @property
    @cache
    def mask_val(self, /) -> int:
        """```py
        0000 1111  0000 1111  0000 1111  0000 1111  0000 1111
             ^^^^       ^^^^       ^^^^       ^^^^       ^^^^
        ```"""
        # works like array of MAXVAL
        return self.mask_array_val * mask(self.bv)

    @property
    @cache
    def mask_array_pad(self, /) -> int:
        """```py
        0001 0000  0001 0000  0001 0000  0001 0000  0001 0000
           ^          ^          ^          ^          ^
        ```"""
        # works like array of MAXVAL+1
        return mask_array(self.len, self.bi) << self.bv

    @property
    @cache
    def mask_pad(self, /) -> int:
        """```py
        1111 0000  1111 0000  1111 0000  1111 0000  1111 0000
        ^^^^       ^^^^       ^^^^       ^^^^       ^^^^
        ```"""
        return self.mask_array_pad * mask(self.bp)


class A:
    """```py
    A(0x_05_04_03_02_01, S(4, 4, 5)):
       0    5     0    4     0    3     0    2     0    1
    0000 0101  0000 0100  0000 0011  0000 0010  0000 0001
    ^^^^       ^^^^       ^^^^       ^^^^       ^^^^      - padding
         ^^^^       ^^^^       ^^^^       ^^^^       ^^^^ - values
    ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^  ^^^^^^^^^ - items
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ - data
            4          3          2          1          0 - items are indexed from least significant to most significant

    requirements:
      - padding bits must be filled with zeros
      - nonzero value width and padding width
      - data is nonnegative
      - if an array dont have enough padding for the operation, then behaviour is unspecified
    ```"""

    __match_args__ = __slots__ = (
        'data',
        's',
    )
    data: int
    s: S

    def __init__(self, data: int, s: S, /) -> None:
        assert data >= 0, f'negative data: {data}'
        assert data == data & s.mask_val, f'padding bits are not cleared'
        self.data = data
        self.s = s

    @classmethod
    def from_const(cls, val: int, s: S, /) -> t.Self:
        # TODO: fast path for val==0?
        assert val >= 0, f'negative value: {val}'
        return cls(s.mask_array_val * val, s)

    @classmethod
    def from_iterable(cls, iterable: t.Iterable[int], s: S, /) -> t.Self:
        # TODO: can be made faster for s.bi==8 case
        vals = iterable
        vals = [bin(x)[2:] for x in vals]  # convert value item to string of bits
        vals = [x.zfill(s.bi) for x in vals]  # add leading zeros and zeros for padding
        vals = reversed(vals)  # first item should be the least significant
        bits = ''.join(vals)
        n = int(bits, 2)
        return cls(n, s)

    # TODO: add from_bytes
    # TODO: add to_bytes

    def __eq__(self, other: object, /) -> bool:
        if other.__class__ is not A:
            return NotImplemented
        if self is other:
            return True
        assert isinstance(other, A), other  # to make typechecker happy
        return self.s == other.s and self.data == other.data

    def __ne__(self, other: object, /) -> bool:
        if other.__class__ is not A:
            return NotImplemented
        if self is other:
            return False
        assert isinstance(other, A), other  # to make typechecker happy
        return self.s != other.s or self.data != other.data

    def _check_shape_compatibility(self, s: S, /) -> None:
        if self.s != s:
            raise ValueError(f'incompatible shapes: {self.s} and {s}')
        return

    def _get_binop_operand(self, other: A | int, /) -> A:
        if isinstance(other, int):
            return self.from_const(other, self.s)
        self._check_shape_compatibility(other.s)
        return other

    def _get_item(self, i: int, /) -> int:
        return self.data >> (i * self.s.bi) & mask(self.s.bi)

    def _get_padval(self, i: int, /) -> tuple[int, int]:
        item = self._get_item(i)
        pad = item >> self.s.bv
        val = item & mask(self.s.bv)
        return pad, val

    def __iter__(self, /) -> t.Iterator[int]:
        # TODO: can be made faster for s.bi==8 case
        nbits = self.s.bi * self.s.len
        bits = f"{self.data:0{nbits}b}"
        for i in range(self.s.len):
            # note: we can't use negative indexing for the range end, because it would fail when i==0
            x = bits[-(i + 1) * self.s.bi + self.s.bp : len(bits) - i * self.s.bi]
            yield int(x, 2)

    def __len__(self, /) -> int:
        return self.s.len

    def __getitem__(self, index: int, /) -> int:
        # TODO: add bounds checking (index<0 or index>=len)
        # WARNING: this is O(len) operation
        _, val = self._get_padval(index)
        return val

    def __str__(self, /) -> str:
        # TODO: make this linear over length
        # (probably dont show padding too because it should always be zero)
        res: list[str] = []
        for i in range(self.s.len):
            pad, val = self._get_padval(i)
            res += [f"{pad:0{self.s.bp}b}_{val:0{self.s.bv}b}"]
        return f"[{", ".join(res)}]"

    def __repr__(self, /) -> str:
        # TODO: what is happening? i have no idea
        bits_per_hex_digit = 4
        hex_digits = self.s.bi * self.s.len / bits_per_hex_digit
        hex_digits = hex_digits.__ceil__()
        return f"{self.__class__.__qualname__}({self.data:#0{hex_digits+2}x}, {self.s})"

    def __add__(self, other: A | int, /) -> t.Self:
        # TODO: fast path for other==0?
        other = self._get_binop_operand(other)
        n1 = self.data
        n2 = other.data
        # perform addition:
        n = n1 + n2
        # fill padding with zeros:
        n &= self.s.mask_val
        return self.__class__(n, self.s)

    def __radd__(self, other: int, /) -> t.Self:
        return self + other

    def __sub__(self, other: A | int, /) -> t.Self:
        # TODO: fast path for other==0?
        other = self._get_binop_operand(other)
        n1 = self.data
        n2 = other.data
        # store 1 in each padding block:
        n1 |= self.s.mask_array_pad
        # perform subtraction:
        n = n1 - n2
        # fill padding with zeros:
        n &= self.s.mask_val
        return self.__class__(n, self.s)

    def __rsub__(self, other: int, /) -> t.Self:
        return -self + other

    def __mul__(self, other: A | int, /) -> t.Self:
        if isinstance(other, int):
            # TODO: fast path for other==0 or other==1?
            # multiplication by constant
            n = self.data
            n *= other
            # fill padding with zeros:
            n &= self.s.mask_val
            return self.__class__(n, self.s)

        self._check_shape_compatibility(other.s)

        n1 = self.data
        n2 = other.data
        n = 0
        # TODO: add comment describing how this works
        for i in range(self.s.bv):
            b = n2
            b >>= i
            b &= self.s.mask_array_val
            b *= mask(self.s.bv)
            b &= n1
            b <<= i
            n += b

        # fill padding with zeros:
        n &= self.s.mask_val
        return self.__class__(n, self.s)

    def __rmul__(self, other: int, /) -> t.Self:
        return self * other

    def __and__(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        n1 = self.data
        n2 = other.data
        n = n1 & n2
        return self.__class__(n, self.s)

    def __rand__(self, other: int, /) -> t.Self:
        return self & other

    def __or__(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        n1 = self.data
        n2 = other.data
        n = n1 | n2
        return self.__class__(n, self.s)

    def __ror__(self, other: int, /) -> t.Self:
        return self | other

    def __xor__(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        n1 = self.data
        n2 = other.data
        n = n1 ^ n2
        return self.__class__(n, self.s)

    def __rxor__(self, other: int, /) -> t.Self:
        return self ^ other

    def __rshift__(self, b: int, /) -> t.Self:
        # TODO: what if b<0?
        # TODO: what if b is too big?
        n = self.data
        n >>= b
        # fill padding with zeros:
        n &= self.s.mask_val
        return self.__class__(n, self.s)

    def __lshift__(self, b: int, /) -> t.Self:
        # TODO: what if b<0?
        # TODO: what if b is too big?
        n = self.data
        n <<= b
        # fill padding with zeros:
        n &= self.s.mask_val
        return self.__class__(n, self.s)

    def __invert__(self, /) -> t.Self:
        n = self.data
        n ^= self.s.mask_val
        return self.__class__(n, self.s)

    def __neg__(self, /) -> t.Self:
        n = self.data
        n = self.s.mask_array_pad - n  # assumes there is a padding?
        # n = self.s.mask_val + self.s.mask_array_val - n # works in the same way
        return self.__class__(n, self.s)

    def is_true(self, /) -> t.Self:
        n = self.data
        # store 1s in each padding block:
        n |= self.s.mask_array_pad
        # subtract 1:
        n -= self.s.mask_array_val
        # isolate the first padding bit: (it will underflow to 0, iff the starting value was 0)
        n >>= self.s.bv
        n &= self.s.mask_array_val
        return self.__class__(n, self.s)

    def is_false(self, /) -> t.Self:
        n = self.is_true().data
        n ^= self.s.mask_array_val
        return self.__class__(n, self.s)

    def eq(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        return (self ^ other).is_false()

    def ne(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        return (self ^ other).is_true()

    def lt(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        # (self-other)<0
        n = (self - other).data
        # move sign bit to the beginning
        n >>= self.s.bv - 1
        n &= self.s.mask_array_val
        return self.__class__(n, self.s)

    def gt(self, other: A | int, /) -> t.Self:
        other = self._get_binop_operand(other)
        # (other-self)<0
        n = (other - self).data
        # move sign bit to the beginning
        n >>= self.s.bv - 1
        n &= self.s.mask_array_val
        return self.__class__(n, self.s)

    def le(self, other: A | int, /) -> t.Self:
        # TODO: maybe reimplement this method from scratch to avoid extra work?
        return self.lt(other) | self.eq(other)

    def ge(self, other: A | int, /) -> t.Self:
        # TODO: maybe reimplement this method from scratch to avoid extra work?
        return self.gt(other) | self.eq(other)
