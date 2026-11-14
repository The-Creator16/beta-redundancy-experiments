"""Scientific table-selection checks against the supplied manuscript values."""
import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("release",ROOT/"experiments/build_release.py")
release=importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

@pytest.fixture(scope="module")
def tables():
    return release.table_data(ROOT/"results/data",ROOT/"results/audit")

def test_document_scopes(tables):
    j,s=tables
    assert list(j)==[f"table_{n}" for n in range(1,6)]
    assert list(s)==[f"table_S{n}" for n in range(1,10)]
    assert len(j["table_2"][0])==10
    assert len(s["table_S1"][0])==25
    assert len(s["table_S7"][0])==15
    assert len(s["table_S8"][0])==3

def test_exact_and_floating_guard_errors_are_distinct(tables):
    j,s=tables
    exact=j["table_5"][0].query("guard_digits == 16")
    floating=s["table_S9"][0].query("guard_digits == 16")
    assert (exact.workspace_error==0).all()
    assert (floating.max_extended_error_no_overflow>0).all()
    assert floating.max_extended_error_no_overflow.max()<2e-13

def test_published_integer_counts(tables):
    j,s=tables
    assert list(j["table_3"][0].semantic)==[0,232,183,29,5]
    assert list(j["table_3"][0].semantic_without_exact)==[0,5,20,2,1]
    assert int(s["table_S5"][0].set_index("system").loc["signed2_naf","single_total_corruptions"])==2796192
    assert int(s["table_S6"][0].set_index("system").loc["signed2_naf","burst_total_corruptions"])==386923068

def test_normalized_deficit_uses_signed_alphabet(tables):
    _,s=tables
    row=s["table_S1"][0].query("system == 'signed2_naf' and W == 8").iloc[0]
    assert row.normalized_deficit==pytest.approx(.3364,abs=5e-5)

def test_figure_membership():
    assert "fig_exp7_truncation_vs_guard" in release.JOURNAL
    assert "fig_exp7_truncation_vs_guard" not in release.SUPPLEMENT
    assert "fig_exp7_mae_vs_guard" in release.SUPPLEMENT
    assert "fig_exp7_mae_vs_guard" not in release.JOURNAL

def test_journal_table_2_matches_printed_rows(tables):
    frame=tables[0]["table_2"][0]
    columns=["structural_detectability_rate","structural_exactness_rate","semantic_survival_rate","NMAE","mean_normalization_steps"]
    expected=[
        ["0.0000","1.0000","0.0000","0.04220","0.00"],
        ["0.0000","1.0000","0.0000","0.03332","0.000"],
        ["0.4892","0.9828","0.0000","0.08785","23.03"],
        ["0.7018","0.9813","0.0232","0.07261","23.021"],
        ["0.4308","0.9606","0.0000","0.08056","0.66"],
        ["0.6024","0.9562","0.0183","0.06964","1.045"],
        ["0.2228","0.9806","0.0000","0.05896","0.26"],
        ["0.3236","0.9806","0.0029","0.04812","0.395"],
        ["0.1194","0.9882","0.0000","0.05141","0.13"],
        ["0.1636","0.9886","0.0005","0.04209","0.178"]]
    assert list(frame.channel)==["Single","Burst"]*5
    actual=[[release.display_cell(c,row[c],row,"journal_table_2") for c in columns]
            for _,row in frame.iterrows()]
    assert actual==expected

def test_journal_table_4_matches_printed_rows(tables):
    frame=tables[0]["table_4"][0]
    expected=[
        ["0.4918","0.5082","0.0000","11.53","21","24"],
        ["0.7525","0.2475","0.0000","23.35","25","25"],
        ["0.3232","0.5010","0.3505","9.91","25","33"],
        ["0.0738","0.4998","0.8565","18.99","42","60"],
        ["0.0167","0.5001","0.9720","25.85","52","69"]]
    actual=[[release.display_cell(c,row[c],row,"journal_table_4") for c in frame.columns[1:]]
            for _,row in frame.iterrows()]
    assert actual==expected

def test_tikz_repair_sequences_preserve_exact_phi_value():
    import re
    def value(word):
        # Horner arithmetic on a+b*phi, using phi^2=phi+1.
        a,b=0,0
        for digit in word: a,b=b+int(digit),a+b
        return a,b
    for filename,expected_flips in [("figure_1.tex",1),("figure_S2.tex",2)]:
        source=(ROOT/"experiments/diagrams"/filename).read_text()
        words=re.findall(r"\d/\{([01](?:,[01]){7})\}",source)
        assert len(words)>=4
        words=[w.replace(",","") for w in words]
        assert sum(a!=b for a,b in zip(words[0],words[1]))==expected_flips
        assert "11" not in words[0] and "11" not in words[-1]
        assert all(value(w)==value(words[1]) for w in words[1:])
        assert value(words[0])!=value(words[-1])
