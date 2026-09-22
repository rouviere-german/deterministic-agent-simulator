"""SimRandom: a seeded, platform-independent PRNG with keyed stream derivation.

Every random choice inside a simulation comes from a :class:`SimRandom`. Two properties matter:

1. **Bit-identical across platforms and Python versions.** The generator is SplitMix64 (Steele,
   Lea and Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014) implemented with
   integer arithmetic only. ``random.Random`` is avoided on purpose: its seeding and some of its
   methods have changed between CPython versions. Floats are produced only as dyadic rationals
   (``k / 2**53``) and compared, never passed through ``log``/``exp``, whose last bit can differ
   between C math libraries on Linux and Windows.
2. **Stable under shrinking.** :meth:`derive` returns an independent stream keyed by labels (for
   example ``("human", approval_token)``). A component that draws from its own keyed stream sees
   the same numbers whether or not other components ran, so deleting a ticket or a fault from a
   failing scenario does not reshuffle every other random decision.
"""

from __future__ import annotations

import hashlib
from collections.abc import MutableSequence, Sequence

__all__ = ["SimRandom"]

_MASK64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_MIX1 = 0xBF58476D1CE4E5B9
_MIX2 = 0x94D049BB133111EB
_TWO_POW_53 = float(1 << 53)


def _encode_label(label: str | int) -> bytes:
    text = str(label)
    # Length-prefixed, so ("ab", "c") and ("a", "bc") derive different streams.
    return f"{len(text)}:{text};".encode()


class SimRandom:
    """A SplitMix64 stream identified by a 32-byte key."""

    __slots__ = ("_key", "_state")

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("SimRandom keys are 32 bytes")
        self._key = key
        self._state = int.from_bytes(key[:8], "big")

    @classmethod
    def from_seed(cls, seed: int) -> SimRandom:
        """The root stream of a simulation seed."""
        return cls(hashlib.sha256(b"das-seed:" + str(seed).encode()).digest())

    @property
    def key(self) -> bytes:
        return self._key

    def derive(self, *labels: str | int) -> SimRandom:
        """An independent stream for ``labels``; unaffected by draws made on this stream."""
        digest = hashlib.sha256(self._key)
        for label in labels:
            digest.update(_encode_label(label))
        return SimRandom(digest.digest())

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN_GAMMA) & _MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
        z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
        return z ^ (z >> 31)

    def below(self, n: int) -> int:
        """Uniform integer in ``[0, n)`` without modulo bias (rejection sampling)."""
        if n <= 0:
            raise ValueError("n must be positive")
        if n == 1:
            return 0
        limit = (1 << 64) - ((1 << 64) % n)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % n

    def randint(self, low: int, high: int) -> int:
        """Uniform integer in ``[low, high]`` (both inclusive)."""
        if high < low:
            raise ValueError("high must be >= low")
        return low + self.below(high - low + 1)

    def random(self) -> float:
        """Uniform float in ``[0, 1)`` with 53 random bits."""
        return (self.next_u64() >> 11) / _TWO_POW_53

    def chance(self, probability: float) -> bool:
        """True with the given probability. 0 never draws; 1 or more always returns True."""
        if probability <= 0.0:
            return False
        if probability >= 1.0:
            return True
        return self.random() < probability

    def choice[T](self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError("cannot choose from an empty sequence")
        return items[self.below(len(items))]

    def weighted_index(self, weights: Sequence[int]) -> int:
        """Index drawn with probability proportional to non-negative integer ``weights``."""
        total = sum(weights)
        if total <= 0 or any(weight < 0 for weight in weights):
            raise ValueError("weights must be non-negative with a positive sum")
        pick = self.below(total)
        for index, weight in enumerate(weights):
            if pick < weight:
                return index
            pick -= weight
        raise AssertionError("unreachable: pick is always below the total weight")

    def shuffle[T](self, items: MutableSequence[T]) -> None:
        """Fisher-Yates shuffle in place."""
        for index in range(len(items) - 1, 0, -1):
            other = self.below(index + 1)
            items[index], items[other] = items[other], items[index]
