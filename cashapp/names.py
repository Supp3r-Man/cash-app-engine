"""Payer-name normalisation and similarity (entity resolution).

Bank payer names are messy: abbreviations (MFG), legal suffixes that come and go (INC, LLC),
fields cut off at 20-ish characters, bank noise words ("ACH CREDIT"), typos. Similarity is built
from three things that can each be read and tested on their own: token coverage in both
directions weighted by how rare each token is, a trigram overlap, and a per-token fuzzy match.
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

LEGAL = {"INC", "INCORPORATED", "LLC", "LTD", "LIMITED", "CORP", "CORPORATION", "CO", "COMPANY", "LP", "LLP", "PLC", "THE", "GMBH"}
NOISE_PREFIX = {"ACH", "CREDIT", "WIRE", "FROM", "PMT", "PAYMENT", "CCD", "PPD", "TRF", "TRANSFER", "EFT", "REMITTANCE", "ORIG", "NAME", "BILL", "PAY"}
ABBREVIATIONS = {
    "INTL": "INTERNATIONAL", "INTERNAT": "INTERNATIONAL", "INTERNATL": "INTERNATIONAL",
    "MFG": "MANUFACTURING", "MANUF": "MANUFACTURING", "MFGR": "MANUFACTURING",
    "DIST": "DISTRIBUTION", "DISTR": "DISTRIBUTION",
    "SVCS": "SERVICES", "SVC": "SERVICES", "SERV": "SERVICES",
    "SYS": "SYSTEMS", "GRP": "GROUP", "IND": "INDUSTRIES", "INDS": "INDUSTRIES",
    "TECH": "TECHNOLOGIES", "TECHS": "TECHNOLOGIES", "ASSOC": "ASSOCIATES",
    "LOGIS": "LOGISTICS", "ELEC": "ELECTRIC", "PKG": "PACKAGING",
    "HLTH": "HEALTHCARE", "HC": "HEALTHCARE", "MATL": "MATERIALS", "MATLS": "MATERIALS",
    "HLDGS": "HOLDINGS", "PTNRS": "PARTNERS", "NATL": "NATIONAL", "MGMT": "MANAGEMENT",
}


def normalize(name: str) -> tuple:
    """Upper-case tokens with bank noise, legal suffixes and abbreviations removed or expanded."""
    text = name.upper().replace("&", " AND ")
    tokens = re.sub(r"[^A-Z0-9]+", " ", text).split()
    while tokens and tokens[0] in NOISE_PREFIX:
        tokens = tokens[1:]
    tokens = [ABBREVIATIONS.get(t, t) for t in tokens if t not in LEGAL]
    return tuple(tokens)


def _within_one(a: str, b: str) -> bool:
    """True when the edit distance (insert, delete, substitute) is at most 1."""
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la > lb:
        a, b, la, lb = b, a, lb, la
    i = 0
    while i < la and a[i] == b[i]:
        i += 1
    if la == lb:
        return a[i + 1:] == b[i + 1:]
    return a[i:] == b[i + 1:]


def token_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 3 and long_.startswith(short):
        return 0.85  # a truncated field: ACME vs ACMEX is not this, but INTERNAT vs INTERNATIONAL is
    if len(short) >= 5 and _within_one(a, b):
        return 0.8  # one typo
    return 0.0


def _trigrams(tokens: tuple) -> set:
    s = " " + " ".join(tokens) + " "
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _dice(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return 2.0 * len(a & b) / (len(a) + len(b))


_SHARED_SCORES: dict = {}  # similarity depends only on the customer master and the payer string, so engine runs share it


class NameIndex:
    """Customer master, prepared once: normalised tokens, rarity weights, trigrams."""

    def __init__(self, customers):
        self.ids = [c.id for c in customers]
        self.tokens = {c.id: normalize(c.legal_name) for c in customers}
        self.trigrams = {cid: _trigrams(t) for cid, t in self.tokens.items()}
        df: dict = {}
        for toks in self.tokens.values():
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = max(1, len(self.ids))
        self._n = n
        self._df = df
        self._cache = _SHARED_SCORES.setdefault(hash(tuple((c.id, c.legal_name) for c in customers)), {})

    def idf(self, token: str) -> float:
        # unseen tokens (a typo, a made-up word) are treated as rare
        return math.log(1.0 + self._n / max(1, self._df.get(token, 0) or 0.5))

    def similarity(self, payer: tuple, cust: tuple, payer_tri: set, cust_tri: set) -> float:
        if not payer or not cust:
            return 0.0

        def coverage(src, dst):
            num = den = 0.0
            for t in src:
                w = self.idf(t)
                den += w
                num += w * max(token_similarity(t, u) for u in dst)
            return num / den if den else 0.0

        pc = coverage(payer, cust)  # how much of what the bank printed is explained by the customer
        cc = coverage(cust, payer)  # how much of the customer's name was printed
        tokens_sim = 0.65 * pc + 0.35 * cc
        return min(1.0, 0.8 * tokens_sim + 0.2 * _dice(payer_tri, cust_tri))

    def scores(self, payer_name: str) -> dict:
        """{customer_id: similarity} for every customer; cached per payer string."""
        hit = self._cache.get(payer_name)
        if hit is None:
            payer = normalize(payer_name)
            tri = _trigrams(payer)
            hit = {cid: self.similarity(payer, self.tokens[cid], tri, self.trigrams[cid]) for cid in self.ids}
            self._cache[payer_name] = hit
        return hit

    def rank(self, payer_name: str, top: int = 5) -> list:
        """[(customer_id, similarity)] best first."""
        s = self.scores(payer_name)
        return sorted(s.items(), key=lambda x: (-x[1], x[0]))[:top]

    def score_for(self, payer_name: str, customer_id: str) -> float:
        return self.scores(payer_name).get(customer_id, 0.0)
