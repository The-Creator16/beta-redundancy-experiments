from redundancy_exp.arithmetic import (
    canonicalize_binary_addition,
    canonicalize_multinacci_addition,
    canonicalize_signed2_naf,
    digitwise_add,
)
from redundancy_exp.systems import make_default_systems
from redundancy_exp.window import DigitWindow


def test_digitwise_add():
    assert digitwise_add((1, 0, 1), (0, 1, 1)) == (1, 1, 2)


def test_binary_canonicalizes_simple_carry():
    window = DigitWindow.balanced(4)
    # positions: 2,1,0,-1.  Raw 0020 means 2 at position 0 -> 1 at position 1.
    result = canonicalize_binary_addition((0, 0, 2, 0), window)
    assert result.success
    assert result.digits == (0, 1, 0, 0)


def test_phi_digit_two_rule_preserves_exact_value_without_boundary():
    window = DigitWindow.balanced(8)
    system = make_default_systems()["phi"]
    raw = (0, 0, 0, 2, 0, 0, 0, 0)
    result = canonicalize_multinacci_addition(raw, window, degree=2)
    assert result.success
    assert system.is_admissible(result.digits)
    assert system.value_key(raw, window) == system.value_key(result.digits, window)


def test_signed2_naf_recode_is_admissible_and_exact_without_overflow():
    window = DigitWindow.balanced(8)
    system = make_default_systems()["signed2_naf"]
    raw = (0, 0, 0, 2, 0, 0, 0, 0)
    result = canonicalize_signed2_naf(raw, window)
    assert result.success
    assert system.is_admissible(result.digits)
    assert system.value_key(raw, window) == system.value_key(result.digits, window)
