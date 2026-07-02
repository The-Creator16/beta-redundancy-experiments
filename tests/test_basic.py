from redundancy_exp.sampling import NAFDP, RunLengthDP
from redundancy_exp.systems import make_default_systems
from redundancy_exp.window import DigitWindow


def test_phi_count_matches_fibonacci_shift_small_lengths():
    # Number of binary strings of length W with no adjacent ones is F_{W+2}.
    expected = {1: 2, 2: 3, 3: 5, 4: 8, 5: 13, 6: 21}
    for W, count in expected.items():
        assert RunLengthDP.build(W, max_run=1).count == count


def test_naf_unranking_generates_admissible_words():
    dp = NAFDP.build(6)
    for rank in range(dp.count):
        word = dp.unrank(rank)
        assert all(not (word[i] != 0 and word[i + 1] != 0) for i in range(len(word) - 1))


def test_clean_samples_are_admissible():
    import numpy as np

    systems = make_default_systems()
    rng = np.random.default_rng(123)
    window = DigitWindow.balanced(8)
    for system in systems.values():
        for _ in range(100):
            word = system.sample_word(window, rng)
            assert system.is_admissible(word)
            norm = system.normalize(word, window)
            assert norm.success
            assert norm.digits == word
            assert system.value_key(word, window) == system.value_key(norm.digits, window)
