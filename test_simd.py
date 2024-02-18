from simd import A, S
import pytest

# shape for testing:
s = S(5, 4, 4)  # 5 items, 4 bits for values, 4 bits for padding
s_small = S(4, 2, 2)

# fmt: off

def test_init() -> None:
    with pytest.raises(Exception):
        A(0, S(-1, 4, 4))
    with pytest.raises(Exception):
        A(0, S(5, 4, 0))
    with pytest.raises(Exception):
        A(0, S(5, 0, 4))
    with pytest.raises(Exception):
        A(-1, S(5, 4, 4))
    with pytest.raises(Exception):
        A(0x_05_04_63_02_01, s)
        #          ^

def test_str() -> None:
    assert (
        str(A(0x_05_04_03_02_01, s)) ==
        '[0000_0001, 0000_0010, 0000_0011, 0000_0100, 0000_0101]'
    )
    assert (
        str(A(0x_02_04_02_04_02, s)) ==
        '[0000_0010, 0000_0100, 0000_0010, 0000_0100, 0000_0010]'
    )

def test_repr() -> None:
    print(repr(A(0x_05_04_03_02_01, s)))
    assert (
        repr(A(0x_05_04_03_02_01, s)) ==
        'A(0x0504030201, S(len=5, bv=4, bp=4))'
    )
    assert (
        repr(A(0x_02_04_02_04_02, s)) ==
        'A(0x0204020402, S(len=5, bv=4, bp=4))'
    )


def test_eq() -> None:
    assert (
        A(0x_02_04_02_04_02, s) ==
        A(0x_02_04_02_04_02, s)
    )
    assert (
        A(0x_02_04_02_04_02, s) !=
        A(0x_02_04_02_04_03, s)
    )
    assert (
        A(0x_02_04_02_04_02, s) !=
        A(0x_02_04_02_04_02, S(5, 3, 5))
    )

def test_from_const() -> None:
    assert (
        A.from_const(5, s) ==
        A(0x_05_05_05_05_05, s)
    )

def test_add() -> None:
    assert (
        A(0x_05_04_03_02_01, s) +
        A(0x_02_04_02_04_02, s) ==
        A(0x_07_08_05_06_03, s)
    )
    assert (
        A(0x_05_04_03_02_01, s) +
        14 ==
        A(0x_03_02_01_00_0f, s)
    )
    assert (
        14 +
        A(0x_05_04_03_02_01, s) ==
        A(0x_03_02_01_00_0f, s)
    )

def test_binops_with_subclasses() -> None:
    class B(A): ...
    a = A(0, s)
    b = B(0, s)
    assert type(a + a) is A
    assert type(b + a) is B
    assert type(a + b) is A  # is this correct?
    assert type(b + b) is B


def test_sub() -> None:
    assert (
        A(0x_05_04_03_02_01, s) -
        A(0x_02_04_02_04_02, s) ==
        A(0x_03_00_01_0e_0f, s)
    )
    assert (
        A(0x_05_04_03_02_01, s) -
        3 ==
        A(0x_02_01_00_0f_0e, s)
    )
    assert (
        3 -
        A(0x_05_04_03_02_01, s) ==
        A(0x_0e_0f_00_01_02, s)
    )


def test_mul() -> None:
    assert (
        A(0x_05_04_03_02_01, s) *
        4 ==
        A(0x_04_00_0c_08_04, s)
    )
    assert (
        4 *
        A(0x_05_04_03_02_01, s) ==
        A(0x_04_00_0c_08_04, s)
    )
    assert (
        A(0x_05_04_03_02_01, s) *
        A(0x_04_02_04_02_04, s) ==
        A(0x_04_08_0c_04_04, s)
    )

def test_neg() -> None:
    assert (
       -A(0x_05_04_03_02_01, s) ==
        A(0x_0b_0c_0d_0e_0f, s)
    )

def test_invert() -> None:
    assert (
       ~A(0x_05_04_03_02_01, s) ==
        A(0x_0a_0b_0c_0d_0e, s)
    )

def test_bit_and() -> None:
    assert (
        A(0b_0000_0001_0010_0011, s_small) &
        A(0b_0011_0001_0010_0000, s_small) ==
        A(0b_0000_0001_0010_0000, s_small)
    )
    assert (
        A(0b_0000_0001_0010_0011, s_small) &
        0b_10 ==
        A(0b_0000_0000_0010_0010, s_small)
    )
    assert (
        0b_10 &
        A(0b_0000_0001_0010_0011, s_small) ==
        A(0b_0000_0000_0010_0010, s_small)
    )

def test_bit_or() -> None:
    assert (
        A(0b_0000_0001_0010_0011, s_small) |
        A(0b_0011_0001_0010_0000, s_small) ==
        A(0b_0011_0001_0010_0011, s_small)
    )
    assert (
        A(0b_0000_0001_0010_0011, s_small) |
        0b_10 ==
        A(0b_0010_0011_0010_0011, s_small)
    )
    assert (
        0b_10 |
        A(0b_0000_0001_0010_0011, s_small) ==
        A(0b_0010_0011_0010_0011, s_small)
    )

def test_bit_xor() -> None:
    assert (
        A(0b_0000_0001_0010_0011, s_small) ^
        A(0b_0011_0001_0010_0000, s_small) ==
        A(0b_0011_0000_0000_0011, s_small)
    )
    assert (
        A(0b_0000_0001_0010_0011, s_small) ^
        0b_10 ==
        A(0b_0010_0011_0000_0001, s_small)
    )
    assert (
        0b_10 ^
        A(0b_0000_0001_0010_0011, s_small) ==
        A(0b_0010_0011_0000_0001, s_small)
    )

def test_bit_shift() -> None:
    assert (
        A(0b_0000_0001_0010_0011, s_small) >>
        1 ==
        A(0b_0000_0000_0001_0001, s_small)
    )
    assert (
        A(0b_0000_0001_0010_0011, s_small) <<
        1 ==
        A(0b_0000_0010_0000_0010, s_small)
    )

def test_to_bool() -> None:
    assert (
        A(0x_0f_00_03_02_01, s).is_true() ==
        A(0x_01_00_01_01_01, s)
    )
    assert (
        A(0x_0f_00_03_02_01, s).is_false() ==
        A(0x_00_01_00_00_00, s)
    )

def test_cmp() -> None:
    assert (
        A(0x_05_04_03_02_01, s).eq(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_00_01_00_00_00, s)
    )
    assert (
        A(0x_05_04_03_02_01, s).ne(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_01_00_01_01_01, s)
    )
    assert (
        A(0x_05_04_03_02_01, s).gt(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_01_00_01_00_00, s)
    )
    assert (
        A(0x_05_04_03_02_01, s).lt(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_00_00_00_01_01, s)
    )
    assert (
        A(0x_05_04_03_02_01, s).ge(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_01_01_01_00_00, s)
    )
    assert (
        A(0x_05_04_03_02_01, s).le(
        A(0x_02_04_02_04_02, s)) ==
        A(0x_00_01_00_01_01, s)
    )

def test_iter() -> None:
    a = A(0x_05_04_03_02_01, s)
    assert list(a) == [1, 2, 3, 4, 5] # note the "reversed" order

    a = A(0x_056_034_012, simd.S(len=3, bv=8, bp=4))
    assert list(a) == [0x12, 0x34, 0x56]

def test_len() -> None:
    a = A(0x_05_04_03_02_01, s)
    assert len(a) == 5

def test_getitem() -> None:
    a = A(0x_05_04_03_02_01, s)
    assert a[0] == 1
    assert a[1] == 2
    assert a[2] == 3
    assert a[3] == 4
    assert a[4] == 5

def test_from_iterable() -> None:
    x = [1, 2, 3, 4, 5]
    assert list(A.from_iterable(x, s)) == x


# fmt: on
