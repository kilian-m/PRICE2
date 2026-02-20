"""Equivalence group construction for Ribo-seq deconvolution.

Builds a splice-aware read equivalence graph from transcript annotations and
maps read start positions to sets of compatible ORFs (ReadGeneratingRegions).
These equivalence groups are the core data structure consumed by the
deconvolution optimiser.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum, auto

import HTSeq

from price2.coverage_model import CoveragePosition
from price2.genomic_features import Transcript


class EquivalenceGroup:
    """A group of reads that are compatible with the same set of ORFs.

    Attributes
    ----------
    length : int
        Number of genomic positions (in codon-space) belonging to this group.
    read_count : int
        Number of observed reads assigned to this group.
    reads : set
        Individual read identifiers assigned to this group.
    """

    length: int
    read_count: int
    reads: set

    def __init__(
        self,
        length: int = 0,
        read_count: int = 0,
    ) -> None:
        self.length = length
        self.read_count = read_count
        self.reads = set()


class NodeType(Enum):
    """Role of a node in the splice or read-equivalence graph."""

    SOURCE = auto()
    INTERNAL = auto()
    SINK = auto()


class Node:
    """A node in a splice graph or read-equivalence DAG.

    Attributes
    ----------
    iv : HTSeq.GenomicInterval
        Genomic interval covered by this node.
    transcripts_positions : dict[Transcript, int]
        Maps each transcript passing through this node to the transcript-
        coordinate position at which the node begins.
    upstream : dict[Node, set[Transcript]]
        Mapping from upstream neighbour nodes to the transcripts that connect
        them to this node.
    downstream : dict[Node, set[Transcript]]
        Mapping from downstream neighbour nodes to the transcripts that connect
        this node to them.
    typ : NodeType
        Role of this node (SOURCE, INTERNAL, or SINK).
    """

    iv: HTSeq.GenomicInterval
    transcripts_positions: dict[Transcript, int]
    upstream: dict[Node, set[Transcript]]
    downstream: dict[Node, set[Transcript]]
    typ: NodeType

    def __init__(self, iv: HTSeq.GenomicInterval, typ: NodeType = NodeType.INTERNAL) -> None:
        self.iv = iv
        self.transcripts_positions: dict[Transcript, int] = {}
        self.upstream: dict[Node, set[Transcript]] = defaultdict(set)
        self.downstream: dict[Node, set[Transcript]] = defaultdict(set)
        self.typ = typ

    @staticmethod
    def source_node() -> Node:
        """Create a sentinel source node for graph traversal."""
        return Node(HTSeq.GenomicInterval("chr1", 0, 0, "+"), NodeType.SOURCE)

    @staticmethod
    def sink_node() -> Node:
        """Create a sentinel sink node for graph traversal."""
        return Node(HTSeq.GenomicInterval("chr1", 1e10, 1e10, "+"), NodeType.SINK)


class EquivalenceGroupIntervals:
    """Transcript positions where reads are compatible with a set of RGRs.

    Stores three ``HTSeq.GenomicArrayOfSets`` — one per reading-frame phase
    (0, 1, 2) — that map codon-coordinate positions on a transcript to the
    set of ``(rgr, frame, CoveragePosition)`` tuples compatible with reads
    starting at those positions.
    """

    def __init__(self) -> None:
        self.intervals: tuple[
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
            HTSeq.GenomicArrayOfSets,
        ] = (
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
        )

    def add_rgr(
        self,
        rgr,
        start: int,
        end: int,
        phase: int | None,
        frame: int | None = None,
        covpos: CoveragePosition = CoveragePosition.middle,
    ) -> None:
        """Register a ReadGeneratingRegion over a transcript position range.

        Parameters
        ----------
        rgr :
            The ReadGeneratingRegion to register.
        start : int
            First transcript position (nucleotide coordinate) of the interval.
        end : int
            One-past-the-last transcript position of the interval.
        phase : int or None
            ``(read_start - transcript_start) % 3`` for coding RGRs;
            ignored (all phases used) for NOISE RGRs.
        frame : int or None
            Reading frame relative to ORF start (0, 1, or 2).
        covpos : CoveragePosition
            Which part of the coverage profile this interval corresponds to.
        """
        if rgr.type == "NOISE":
            phases = [0, 1, 2]
        else:
            phases = [phase]
        start_phase = start % 3
        end_phase = end % 3
        for phase in phases:
            if phase >= start_phase:
                s = start + (phase - start_phase)
            else:
                s = start + (phase - start_phase) + 3
            if phase >= end_phase:
                e = end + (phase - end_phase)
            else:
                e = end + (phase - end_phase) + 3
            iv = HTSeq.GenomicInterval(".", s // 3, e // 3, ".")
            self.intervals[phase][iv] += (rgr, frame, covpos)

    def get_egs_dict(self, read_length: int, oua: bool) -> dict:
        """Convert internal intervals to a dict of EquivalenceGroups.

        Parameters
        ----------
        read_length : int
            Read length associated with these equivalence groups.
        oua : bool
            Whether this is for the upstream-annotated (True) or downstream
            (False) cleavage model variant.

        Returns
        -------
        dict
            Maps ``(frozenset_of_rgr_frame_covpos, read_length, oua)`` keys to
            :class:`EquivalenceGroup` values.
        """
        egs: dict = defaultdict(EquivalenceGroup)
        for phase in range(3):
            for step_iv, step_set in self.intervals[phase].steps():
                if not step_set:
                    continue
                rgr_frame_covpos = frozenset(step_set)
                egs[(rgr_frame_covpos, read_length, oua)].length += step_iv.length
        return egs


def make_splice_graph(transcripts: HTSeq.GenomicArrayOfSets) -> Node:
    """Build a splice graph (DAG) from a genomic array of transcript sets.

    Each node in the returned DAG corresponds to a contiguous exonic block
    (a step in *transcripts*) and stores the transcript-coordinate offset at
    which that block begins for every transcript that passes through it.

    Parameters
    ----------
    transcripts : HTSeq.GenomicArrayOfSets
        Genomic array where each step maps to the set of transcripts covering
        that interval.

    Returns
    -------
    Node
        The source sentinel node of the splice graph.
    """
    source = Node.source_node()
    transcript_lengths = defaultdict(int)
    prev_nodes = dict()

    steps = list(transcripts.steps())
    c_p, c_m = 0, 0
    for step in steps:
        if step[0].strand == "+":
            c_p += 1
        else:
            c_m += 1
        if c_p == 2:
            strand = "+"
            break
        if c_m == 2:
            strand = "-"
            break

    if strand == "-":
        steps = steps[::-1]

    for step_iv, transcript_set in steps:
        if not transcript_set:
            continue
        node = Node(step_iv)
        for transcript in transcript_set:
            source.transcripts_positions[transcript] = 0
            node.transcripts_positions[transcript] = transcript_lengths[transcript]
            transcript_lengths[transcript] += step_iv.length
            if transcript not in prev_nodes:
                node.upstream[source].add(transcript)
                source.downstream[node].add(transcript)
            else:
                node.upstream[prev_nodes[transcript]].add(transcript)
                prev_nodes[transcript].downstream[node].add(transcript)
            prev_nodes[transcript] = node

    sink = Node.sink_node()
    for transcript, node in prev_nodes.items():
        node.downstream[sink].add(transcript)
        sink.upstream[node].add(transcript)
    return source


def make_read_equivalence_node(
    start: int,
    length: int,
    transcripts_positions: dict[Transcript, int],
) -> Node:
    """Create a read-equivalence node covering a fixed genomic range.

    Parameters
    ----------
    start : int
        Genomic start position of the node interval.
    length : int
        Length of the interval in nucleotides.
    transcripts_positions : dict[Transcript, int]
        Transcript-to-transcript-coordinate mapping for this node.

    Returns
    -------
    Node
        A new INTERNAL node with the given interval and transcript positions.
    """
    gi = HTSeq.GenomicInterval("chr1", start, start + length, "+")
    node = Node(gi)
    node.transcripts_positions = transcripts_positions
    return node


def recur_shift_read_end(
    root_splice_node: Node,
    current_splice_node: Node,
    read_length: int,
    cumulated_length: int,
    intersected_transcripts_positions: dict[Transcript, int],
) -> set[Node]:
    """Recursively extend a read to its 3' end across splice junctions.

    Walks downstream through the splice graph until *read_length* nucleotides
    have been accumulated, then delegates to :func:`recur_shift_read_start` to
    build the corresponding read-equivalence node.

    Parameters
    ----------
    root_splice_node : Node
        The splice-graph node at which the current read starts.
    current_splice_node : Node
        The splice-graph node currently being processed in the recursion.
    read_length : int
        Total length of the read in nucleotides.
    cumulated_length : int
        Nucleotides accumulated so far while traversing downstream.
    intersected_transcripts_positions : dict[Transcript, int]
        Transcripts still compatible with the read and their current
        transcript-coordinate positions.

    Returns
    -------
    set[Node]
        Set of read-equivalence nodes representing all valid end positions.
    """
    # 1. case current_node is sink
    # return
    if current_splice_node.typ == NodeType.SINK:
        return set()
    if current_splice_node.typ == NodeType.SOURCE:
        node = Node.source_node()
        node.transcripts_positions = intersected_transcripts_positions
        return {node}

    # 2. case read_length is reached in current_node
    # recursion end, call recur_shift_read_start
    elif cumulated_length + current_splice_node.iv.length >= read_length:
        read_end_position = (
            current_splice_node.iv.start + read_length - cumulated_length
        )
        return {
            recur_shift_read_start(
                root_splice_node,
                root_splice_node.iv.start,
                current_splice_node,
                read_end_position,
                intersected_transcripts_positions,
            )
        }

    # 3. case read_length is not reached in current_node
    # recurse
    else:
        cumulated_length += current_splice_node.iv.length
        s = set()
        for node, transcripts in current_splice_node.downstream.items():
            itp = {
                t: i
                for t, i in intersected_transcripts_positions.items()
                if t in transcripts
            }
            s |= recur_shift_read_end(
                root_splice_node,
                node,
                read_length,
                cumulated_length,
                itp,
            )
        return s


def recur_shift_read_start(
    root_splice_node: Node,
    read_start_position: int,
    current_splice_node: Node,
    read_end_position: int,
    intersected_transcripts_positions: dict[Transcript, int],
) -> Node:
    """Recursively build a read-equivalence node from a fixed read end.

    Starting from the splice-graph node that contains the read's 3' end,
    walks upstream to construct a chain of read-equivalence nodes whose
    combined length equals the read length.

    Parameters
    ----------
    root_splice_node : Node
        The splice-graph node at which the current read starts.
    read_start_position : int
        Genomic position of the read's 5' end.
    current_splice_node : Node
        The splice-graph node currently being processed in the recursion.
    read_end_position : int
        Genomic position of the read's 3' end within *current_splice_node*.
    intersected_transcripts_positions : dict[Transcript, int]
        Transcripts still compatible with the read and their current
        transcript-coordinate positions.

    Returns
    -------
    Node
        The head read-equivalence node for this read start position.
    """
    # 1. case current node is sink
    # return
    if current_splice_node.typ == NodeType.SINK:
        return Node.sink_node()

    # 2. case read start position reaches the end of the root node
    # recursion end
    elif (root_splice_node.iv.end - read_start_position) < (
        current_splice_node.iv.end - read_end_position
    ):
        current_node_length = root_splice_node.iv.end - read_start_position  # ?
        new_node = make_read_equivalence_node(
            read_start_position,
            current_node_length,
            {k: v for k, v in intersected_transcripts_positions.items()},
        )
        return new_node

    # 3. case read end position reaches the end of the current node
    # recurse
    else:
        current_node_length = current_splice_node.iv.end - read_end_position  # +1 # ?
        if current_splice_node == root_splice_node:
            current_node_length += 1
        new_node = make_read_equivalence_node(
            read_start_position,
            current_node_length,
            intersected_transcripts_positions,
        )
        for node, transcripts in current_splice_node.downstream.items():
            if tr_subset := (intersected_transcripts_positions.keys() & transcripts):
                child_node = recur_shift_read_start(
                    root_splice_node,
                    read_start_position
                    + current_node_length,  # (current_splice_node.iv.end - read_end_position),
                    node,
                    node.iv.start,
                    {
                        t: intersected_transcripts_positions[t] + current_node_length
                        for t in tr_subset
                    },
                )
                new_node.downstream[child_node] = tr_subset
                child_node.upstream[new_node] = tr_subset

        return new_node


def collapse_dag_chains(source_node: Node) -> None:
    """Merge unbranched chains in a DAG into single nodes in-place.

    Traverses the DAG starting from *source_node* and collapses any sequence
    of consecutive nodes where each node has exactly one downstream and one
    upstream neighbour into a single node with an extended interval.

    Parameters
    ----------
    source_node : Node
        Entry point of the DAG to collapse.
    """
    visited: set[Node] = set()
    stack: list[Node] = [source_node]

    while stack:
        current_node = stack.pop()
        if current_node in visited:
            continue
        visited.add(current_node)

        while (
            len(current_node.downstream) == 1
            and (child := next(iter(current_node.downstream))).typ != NodeType.SINK
            and len(child.upstream) == 1
            and child.iv.start == current_node.iv.end
        ):
            tr = current_node.downstream[child]
            for grandchild in child.downstream:
                del grandchild.upstream[child]
                grandchild.upstream[current_node] = tr
            current_node.iv = HTSeq.GenomicInterval(
                current_node.iv.chrom,
                current_node.iv.start,
                child.iv.end,
                current_node.iv.strand,
            )
            current_node.downstream = child.downstream

        for child in current_node.downstream:
            if child not in visited:
                stack.append(child)


def get_tr_2_leaf(
    nodes: set[Node],
) -> dict[Transcript, tuple[Node, int]]:
    """Return the last (leaf) node and end position for every transcript.

    Performs a depth-first traversal of the subgraph rooted at each node in
    *nodes* and records, for every transcript, the deepest node it reaches and
    the transcript-coordinate position at that node's end.

    Parameters
    ----------
    nodes : set[Node]
        Root nodes from which to start the traversal.

    Returns
    -------
    dict[Transcript, tuple[Node, int]]
        Maps each transcript to ``(leaf_node, end_transcript_position)``.
    """
    tr_2_leaf: dict[Transcript, tuple[Node, int]] = {}

    def dfs(node: Node) -> None:
        for tr in node.transcripts_positions.keys():
            tr_2_leaf[tr] = (node, node.transcripts_positions[tr] + node.iv.length)
        for child in node.downstream:
            dfs(child)

    for node in nodes:
        dfs(node)
    return tr_2_leaf


def connect(
    tr_2_leaf: dict[Transcript, tuple[Node, int]],
    child_node_set: set[Node],
) -> None:
    """Connect leaf nodes from one splice-graph block to the next block's heads.

    For every node in *child_node_set*, checks whether any of its transcripts
    end exactly where a leaf node in *tr_2_leaf* ends, and if so adds the
    appropriate directed edges.

    Parameters
    ----------
    tr_2_leaf : dict[Transcript, tuple[Node, int]]
        As returned by :func:`get_tr_2_leaf` for the preceding block.
    child_node_set : set[Node]
        Head nodes of the next block to connect to.
    """
    for child in child_node_set:
        for tr, pos in child.transcripts_positions.items():
            if tr not in tr_2_leaf:
                continue
            if tr_2_leaf[tr][1] == pos:
                tr_2_leaf[tr][0].downstream[child].add(tr)
                child.upstream[tr_2_leaf[tr][0]].add(tr)


def traverse_dag(splice_graph: Node, read_length: int) -> Node:
    """Build a read-equivalence DAG for a given read length.

    Traverses the splice graph and, for each splice node, constructs the set
    of read-equivalence nodes representing all possible read placements of
    length *read_length* that start within that node.  The resulting DAG is
    collapsed with :func:`collapse_dag_chains`.

    Parameters
    ----------
    splice_graph : Node
        Source node of the splice graph (as returned by
        :func:`make_splice_graph`).
    read_length : int
        Length of reads to model.

    Returns
    -------
    Node
        Source node of the collapsed read-equivalence DAG.
    """
    d: dict[Node, set[Node]] = {}
    visited: set[Node] = set()

    def dfs(splice_node: Node) -> None:
        if splice_node in visited:
            return

        d[splice_node] = recur_shift_read_end(
            splice_node,
            splice_node,
            read_length,
            0,
            splice_node.transcripts_positions,
        )
        transcript_2_leaf = get_tr_2_leaf(d[splice_node])

        visited.add(splice_node)
        for child in splice_node.downstream:
            dfs(child)
            connect(transcript_2_leaf, d[child])

    dfs(splice_graph)

    read_equivalence_graph = next(iter(d[splice_graph]))
    collapse_dag_chains(read_equivalence_graph)

    return read_equivalence_graph


def get_equivalence_groups_dict(
    source_node: Node,
    egis: dict[Transcript, EquivalenceGroupIntervals],
    read_length: int,
    oua: bool,
) -> dict:
    """Aggregate equivalence groups over the full read-equivalence DAG.

    Walks all INTERNAL nodes in the read-equivalence DAG and collects, for
    each node, the partial equivalence groups contributed by its transcripts'
    coverage intervals.  Groups with matching keys are merged by summing their
    lengths.

    Parameters
    ----------
    source_node : Node
        Source node of the read-equivalence DAG.
    egis : dict[Transcript, EquivalenceGroupIntervals]
        Per-transcript equivalence-group intervals for the current read length
        and oua flag.
    read_length : int
        Read length for which these groups are computed.
    oua : bool
        Whether this is for the upstream-annotated cleavage model variant.

    Returns
    -------
    dict
        Maps equivalence-group keys to :class:`EquivalenceGroup` instances.
    """
    egs_dict: dict = {}
    stack: list[Node] = []
    stack.append(source_node)
    visited = set()

    while stack:
        current_node = stack.pop()
        if current_node in visited:
            continue
        visited.add(current_node)
        stack.extend(current_node.downstream.keys())
        if current_node.typ == NodeType.SINK or current_node.typ == NodeType.SOURCE:
            continue

        node_eg_dict = get_sub_intervals(
            [egis[tr] for tr in current_node.transcripts_positions.keys()],
            [
                current_node.transcripts_positions[tr]
                for tr in current_node.transcripts_positions.keys()
            ],
            current_node.iv.length,
        ).get_egs_dict(read_length, oua)

        for k, v in node_eg_dict.items():
            try:
                egs_dict[k].length += v.length
            except KeyError:
                egs_dict[k] = v

    return egs_dict


def make_equivalence_groups(loc, runs: list) -> dict:
    """Compute all equivalence groups for a locus across all runs.

    For every run and every read length present in its cleavage model, builds
    the read-equivalence DAG and collects equivalence groups for each
    transcript in the locus.

    Parameters
    ----------
    loc :
        A locus object with ``transcript_intervals`` (GenomicArrayOfSets) and
        ``transcripts`` (list of Transcript).
    runs : list
        List of RiboSeqRun objects, each providing a cleavage model.

    Returns
    -------
    dict
        Maps each run to a dict of equivalence-group key →
        :class:`EquivalenceGroup`.
    """
    egs: dict = {}
    splice_graph = make_splice_graph(loc.transcript_intervals)

    for run in runs:
        egs[run] = {}
        cleavage_model = run.cleavage_model
        for read_length in run.cleavage_model.non_zero_lengths:
            read_equivalence_graph = traverse_dag(splice_graph, read_length)
            for oua in [True, False]:
                egis = {
                    tr: make_equivalence_intervals(tr, cleavage_model, read_length, oua)
                    for tr in loc.transcripts
                }

                for k, v in get_equivalence_groups_dict(
                    read_equivalence_graph, egis, read_length, oua
                ).items():

                    try:
                        egs[run][k].length += v.length
                    except KeyError:
                        egs[run][k] = v

    return egs


def get_sub_intervals(
    egis: list[EquivalenceGroupIntervals],
    starts: list[int],
    length: int,
) -> EquivalenceGroupIntervals:
    """Extract and merge sub-intervals from multiple transcript EGIs.

    For each ``(egi, start)`` pair, slices out the portion of *egi* that
    corresponds to transcript positions ``[start, start + length)`` and
    assembles the pieces into a single :class:`EquivalenceGroupIntervals`
    aligned to position 0.

    Parameters
    ----------
    egis : list[EquivalenceGroupIntervals]
        Equivalence-group intervals for each transcript passing through a node.
    starts : list[int]
        Transcript-coordinate start positions for each transcript within the
        node (parallel to *egis*).
    length : int
        Length of the node interval in nucleotides.

    Returns
    -------
    EquivalenceGroupIntervals
        Combined equivalence-group intervals aligned to position 0.
    """
    result_intervals = (
        HTSeq.GenomicArrayOfSets("auto", stranded=False),
        HTSeq.GenomicArrayOfSets("auto", stranded=False),
        HTSeq.GenomicArrayOfSets("auto", stranded=False),
    )

    for egi, start in zip(egis, starts):
        end = start + length
        s_q, s_r = divmod(start, 3)
        e_q, e_r = divmod(end, 3)
        for i, egi_iv in enumerate(egi.intervals):
            p = (i - s_r) % 3
            s = s_q if s_r <= i else s_q + 1
            e = e_q + 1 if e_r > i else e_q
            iv = HTSeq.GenomicInterval(".", s, e, ".")
            l = 0
            try:
                for step_iv, step_set in egi_iv[iv].steps():
                    if not step_set:
                        l += step_iv.length
                        continue
                    siv = HTSeq.GenomicInterval(
                        ".",
                        l,
                        l + step_iv.length,
                        ".",
                    )
                    for s in step_set:
                        result_intervals[p][siv] += s
                    l += step_iv.length
            except IndexError:
                pass

    egi = EquivalenceGroupIntervals()
    egi.intervals = result_intervals
    return egi


def make_equivalence_intervals(
    transcript: Transcript,
    cleavage_model,
    read_length: int,
    oua: bool,
) -> EquivalenceGroupIntervals:
    """Build equivalence-group intervals for one transcript.

    Uses the cleavage model to determine, for each ReadGeneratingRegion on
    *transcript*, which transcript positions can produce a read of
    *read_length* and in which frame/coverage-position category.

    Parameters
    ----------
    transcript : Transcript
        The transcript for which to compute intervals.
    cleavage_model :
        Cleavage model providing ``get_dist_to_orf_start`` and
        ``get_dist_to_orf_end`` for each read length, oua flag, and frame.
    read_length : int
        Read length to model.
    oua : bool
        Whether to use the upstream-annotated cleavage model variant.

    Returns
    -------
    EquivalenceGroupIntervals
        Populated intervals for assignment of reads to RGRs on this transcript.
    """
    egi = EquivalenceGroupIntervals()
    for rgr in transcript.rgr_set:
        if rgr.type == "NOISE":
            try:
                start = max(
                    0,
                    rgr.iv_on_transcript[0]
                    + cleavage_model.get_dist_to_orf_start(read_length, oua, None),
                )
                end = max(
                    3,
                    min(
                        len(transcript.exons) - read_length + 1,
                        rgr.iv_on_transcript[1]
                        + cleavage_model.get_dist_to_orf_end(read_length, oua, None),
                    ),
                )
            except KeyError:
                continue
            try:
                egi.add_rgr(rgr, start, end, None)
            except (IndexError, ValueError):
                pass

        else:
            for frame in [0, 1, 2]:  # frame relative to ORF start
                try:
                    rsos = cleavage_model.get_dist_to_orf_start(
                        read_length, oua, frame
                    )  # read_start_to_orf_start
                    rsoe = cleavage_model.get_dist_to_orf_end(
                        read_length, oua, frame
                    )  # read_start_to_orf_end
                    phase = (
                        rgr.iv_on_transcript[0] + rsos
                    ) % 3  # phase relative to transcript start
                except KeyError:
                    continue

                # CoveragePosition.start
                start = max(0, rgr.iv_on_transcript[0] + rsos)
                end = max(0, rgr.iv_on_transcript[0] + 3 + rsoe)
                try:
                    egi.add_rgr(rgr, start, end, phase, frame, CoveragePosition.start)
                except (IndexError, ValueError):
                    pass

                # CoveragePosition.middle
                start = max(0, rgr.iv_on_transcript[0] + 3 + rsos)
                end = min(
                    transcript.iv.length - read_length + 1,
                    rgr.iv_on_transcript[1] - 3 + rsoe,
                )
                try:
                    egi.add_rgr(rgr, start, end, phase, frame, CoveragePosition.middle)
                except (IndexError, ValueError):
                    pass

                # CoveragePosition.stop
                start = max(0, rgr.iv_on_transcript[1] - 3 + rsos)
                end = min(
                    transcript.iv.length - read_length + 1,
                    rgr.iv_on_transcript[1] + rsoe,
                )
                try:
                    egi.add_rgr(rgr, start, end, phase, frame, CoveragePosition.stop)
                except (IndexError, ValueError):
                    pass
    return egi
