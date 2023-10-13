import pytest

from price2.cleavage_model import CleavageEstimator
from price2.reference_annotation import ReferenceAnnotation

import HTSeq

reference_annotation = ReferenceAnnotation('tests/test.gtf')

bam_reader = HTSeq.BAM_Reader('tests/test.bam')
bam_dict = {x.read.name:x for x in bam_reader}



def test_non_unique():
    alignments = [bam_dict['n001']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 0


def test_frame_0():
    alignments = [bam_dict['n002']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[30,0,0,0] == 1

def test_frame_0_2():
    alignments = [bam_dict['n006']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[31,0,0,0] == 1

def test_frame_1():
    alignments = [bam_dict['n003']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[30,1,0,0] == 1

def test_frame_2():
    alignments = [bam_dict['n004']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[30,2,0,0] == 1

def test_split_read():
    alignments = [bam_dict['n005']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[29,0,0,0] == 1

def test_reverse_invalid():
    alignments = [bam_dict['n007']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 0

def test_reverse_valid():
    alignments = [bam_dict['n008']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 1
    assert ce.table[31,0,0,0] == 1

############################################################

def test_overlapping_cds_end():
    alignments = [bam_dict['r001']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)
    print(ce.table[:,:,0,0])
    assert ce.table.sum() == 0


def test_outside_cds():
    alignments = [bam_dict['r006']]
    ce = CleavageEstimator(1)

    ce.fill_table(reference_annotation, alignments)

    assert ce.table.sum() == 0