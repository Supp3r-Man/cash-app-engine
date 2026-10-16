import random

from cashapp.names import NameIndex, normalize, token_similarity
from cashapp.refs import RefIndex, edit_distance_le1, extract_tokens
from cashapp.schema import Customer
from helpers import invoice


def test_normalize_drops_suffix_noise_and_expands_abbreviations():
    assert normalize("ACH CREDIT Acme Mfg, Inc.") == ("ACME", "MANUFACTURING")
    assert normalize("THE ACME MANUFACTURING CO") == ("ACME", "MANUFACTURING")
    assert normalize("Smith & Sons LLC") == ("SMITH", "AND", "SONS")


def test_token_similarity_tiers():
    assert token_similarity("ACME", "ACME") == 1.0
    assert token_similarity("INTERNAT", "INTERNATIONAL") == 0.85  # truncated bank field
    assert token_similarity("LOGISTCS", "LOGISTICS") == 0.8  # one typo
    assert token_similarity("ACME", "ATLAS") == 0.0


def _index():
    return NameIndex([Customer("C1", "Acme Manufacturing Inc"), Customer("C2", "Acme Industries LLC"),
                      Customer("C3", "Birch Logistics Corp")])


def test_name_variants_resolve_to_the_right_customer():
    ix = _index()
    for variant, want in [("ACME MFG", "C1"), ("ACME MANUFACTURING I", "C1"), ("ACH CREDIT ACME INDUSTRIES", "C2"), ("BIRCH LOGISTCS", "C3")]:
        top = ix.rank(variant, 2)
        assert top[0][0] == want and top[0][1] > top[1][1] + 0.2, variant


def test_a_bare_first_word_is_ambiguous_and_a_stranger_matches_nothing():
    ix = _index()
    (a, sa), (b, sb) = ix.rank("ACME", 2)
    assert {a, b} == {"C1", "C2"} and abs(sa - sb) < 0.1
    assert ix.rank("ZEPHYR VENTURES", 1)[0][1] < 0.2


def test_extract_tokens_ignores_money_dates_and_trace_ids():
    toks = extract_tokens("ACH-0042918837 ; INV 4821937 1,250.00 ; PAID 2025-09-03 ; INVOICE 4821944")
    assert [t.digits for t in toks] == ["4821937", "4821944"]


def test_extract_tokens_repairs_ocr_letters_and_prefixes():
    assert [t.digits for t in extract_tokens("INV# 48219O7")] == ["4821907"]
    assert [t.digits for t in extract_tokens("INV4821937")] == ["4821937"]
    assert extract_tokens("THANK YOU") == []


def test_edit_distance_le1_cases():
    assert edit_distance_le1("4821937", "4821937")
    assert edit_distance_le1("4821937", "4821397")  # adjacent swap
    assert edit_distance_le1("4821937", "482937")  # deletion
    assert edit_distance_le1("4821937", "4821837")  # substitution
    assert not edit_distance_le1("4821937", "4821397" + "0")
    assert not edit_distance_le1("4821937", "4812397")  # two edits


def test_ref_index_tiers():
    invs = [invoice("I1", "4821937", 1000, po="884213"), invoice("I2", "1234567", 1000)]
    ix = RefIndex(invs)

    def best(text):
        hits = ix.lookup_all(extract_tokens(text))
        return {k: (v.kind, v.sim) for k, v in hits.items()}

    assert best("4821937") == {"I1": ("exact", 1.0)}
    assert best("PO 884213") == {"I1": ("po", 0.9)}
    assert best("4821397") == {"I1": ("fuzzy", 0.75)}
    assert best("21937") == {"I1": ("suffix", 0.70)}
    assert best("9999999") == {}
    assert RefIndex(invs, fuzzy=False).lookup_all(extract_tokens("4821397")) == {}


def test_fuzzy_index_agrees_with_brute_force():
    rng = random.Random(1)
    invs = [invoice(f"I{k}", str(rng.randrange(1_000_000, 9_999_999)), 100) for k in range(300)]
    ix = RefIndex(invs)
    for _ in range(300):
        d = str(rng.randrange(1_000_000, 9_999_999)) if rng.random() < 0.3 else invs[rng.randrange(300)].digits
        if rng.random() < 0.6:
            i = rng.randrange(len(d))
            d = d[:i] + str(rng.randrange(10)) + d[i + 1:]
        from cashapp.refs import RefToken
        got = {k for k, h in ix.lookup(RefToken(d, d)).items() if h.kind in ("exact", "fuzzy")}
        want = {i.id for i in invs if edit_distance_le1(d, i.digits)}
        assert got == want
