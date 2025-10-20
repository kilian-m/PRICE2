from enum import Enum, auto
import HTSeq
from collections import defaultdict

from price2.genomic_features import Transcript
from price2.coverage_model import CoveragePosition


class EquivalenceGroup:
    length: int
    read_count: int

    def __init__(
        self,
        length: int = 0,
        read_count: int = 0,
    ) -> None:
        self.length = length
        self.read_count = read_count
        self.reads = set()


class NodeType(Enum):
    SOURCE = auto()
    INTERNAL = auto()
    SINK = auto()


class Node:
    iv: HTSeq.GenomicInterval
    transcripts_positions: dict[Transcript, int]
    upstream: dict["Node", set[Transcript]]
    downstream: dict["Node", set[Transcript]]
    typ: NodeType

    def __init__(self, iv: HTSeq.GenomicInterval, typ=NodeType.INTERNAL):
        self.iv = iv
        self.transcripts_positions = dict()  # transcript -> transcript_position: int
        self.upstream = defaultdict(set)  # node -> set[transcript]
        self.downstream = defaultdict(set)  # node -> set[transcript]
        self.typ = typ

    def source_node():
        return Node(HTSeq.GenomicInterval("chr1", 0, 0, "+"), NodeType.SOURCE)

    def sink_node():
        return Node(HTSeq.GenomicInterval("chr1", 1e10, 1e10, "+"), NodeType.SINK)


class EquivalenceGroupIntervals:
    # represents positions on a transcript where reads that start there are compatible with rgrs is a certain frame
    def __init__(self):
        self.intervals = (
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
            HTSeq.GenomicArrayOfSets("auto", stranded=False),
        )

    def add_rgr(
        self, rgr, start, end, phase, frame=None, covpos=CoveragePosition.middle
    ):
        # phase: (read_start - transcript_start) % 3
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

    def get_egs_dict(self, read_length, oua):
        egs = defaultdict(EquivalenceGroup)
        for phase in range(3):
            for step_iv, step_set in self.intervals[phase].steps():
                if not step_set:
                    continue
                rgr_frame_covpos = frozenset(step_set)
                egs[(rgr_frame_covpos, read_length, oua)].length += step_iv.length
        return egs


def make_splice_graph(transcripts: HTSeq.GenomicArrayOfSets) -> Node:
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
    start,
    length,
    transcripts_positions,
):
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


def collapse_dag_chains(source_node):
    visited = set()
    stack = [source_node]

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


def get_tr_2_leaf(nodes):
    tr_2_leaf = {}

    def dfs(node):
        for tr in node.transcripts_positions.keys():
            tr_2_leaf[tr] = (node, node.transcripts_positions[tr] + node.iv.length)
        for child in node.downstream:
            dfs(child)

    for node in nodes:
        dfs(node)
    return tr_2_leaf


def connect(tr_2_leaf, child_node_set):
    for child in child_node_set:
        for tr, pos in child.transcripts_positions.items():
            if not tr in tr_2_leaf:
                continue
            if tr_2_leaf[tr][1] == pos:
                tr_2_leaf[tr][0].downstream[child].add(tr)
                child.upstream[tr_2_leaf[tr][0]].add(tr)


def traverse_dag(splice_graph, read_length) -> dict:
    d = {}
    visited = set()

    def dfs(splice_node):
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


def bla(source_node, egis, read_length, oua):
    egs_dict = {}
    stack = []
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


def make_equivalence_groups(loc, runs):
    egs = {}
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

                for k, v in bla(read_equivalence_graph, egis, read_length, oua).items():

                    try:
                        egs[run][k].length += v.length
                    except KeyError:
                        egs[run][k] = v

    return egs


# put in multiple egi with corresponding starts and node length
# generate combined egi
def get_sub_intervals(egis, starts, length):
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


def make_equivalence_intervals(transcript, cleavage_model, read_length, oua):
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
