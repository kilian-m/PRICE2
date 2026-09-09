"""Where a codon sits in its ORF."""

from enum import Enum


class CoveragePosition(Enum):
    """Position category of a codon relative to its ORF.

    The integer values are load-bearing: :mod:`price2.locus` packs a codon's
    ``frame * 3 + value`` into one byte, and the coverage factors are indexed
    ``[start, middle, stop]`` in that order.

    Members
    -------
    start
        The start (initiator) codon.
    middle
        Any codon in the ORF body.
    stop
        The codon immediately upstream of the stop codon.
    """

    start = 0
    middle = 1
    stop = 2
