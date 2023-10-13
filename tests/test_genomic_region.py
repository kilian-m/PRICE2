import pytest

from price2.genomic_region import GenomicRegion
import HTSeq

##########################################
# test induce

cds_ivs = [
    HTSeq.GenomicInterval( "chr1", 100, 110, "+" ),
    HTSeq.GenomicInterval( "chr1", 120, 130, "+" ),
    HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
    HTSeq.GenomicInterval( "chr1", 160, 170, "+" ),
    HTSeq.GenomicInterval( "chr1", 180, 190, "+" ),
]

cds = GenomicRegion(cds_ivs)


def test_genomic_region_1():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 103, 108, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (3, 8)


def test_genomic_region_2():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 105, 110, "+" ),
        HTSeq.GenomicInterval( "chr1", 120, 126, "+" ),
        ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (5, 16)


def test_genomic_region_3():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 143, 148, "+" ),
        ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (23,28)


def test_genomic_region_4():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 123, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 166, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (13, 36)

def test_genomic_region_5():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 123, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 170, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (13, 40)

def test_genomic_region_6():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 100, 110, "+" ),
        HTSeq.GenomicInterval( "chr1", 120, 125, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (0, 15)

def test_genomic_region_7():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 165, 170, "+" ),
        HTSeq.GenomicInterval( "chr1", 180, 190, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (35,50)

def test_genomic_region_8():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 100, 110, "+" ),
        HTSeq.GenomicInterval( "chr1", 120, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 170, "+" ),
        HTSeq.GenomicInterval( "chr1", 180, 190, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (0, 50)

def test_genomic_region_9():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 90, 110, "+" ),
        HTSeq.GenomicInterval( "chr1", 120, 125, "+" ),
    ]
    read = GenomicRegion(ivs)
    #assert cds.induce(read) == False
    assert cds.induce(read) == (-10, 15)

def test_genomic_region_10():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 165, 170, "+" ),
        HTSeq.GenomicInterval( "chr1", 180, 195, "+" ),
    ]
    read = GenomicRegion(ivs)
    #assert cds.induce(read) == False
    assert cds.induce(read) == (35, 55)
    
def test_genomic_region_11():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 165, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (20, 35)

def test_genomic_region_12():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (20, 30)

def test_genomic_region_13():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 125, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 133, 138, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)

def test_genomic_region_14():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 70, 80, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (-30, -20)

def test_genomic_region_15():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 200, 210, "+" ),
    ]
    read = GenomicRegion(ivs)
    assert cds.induce(read) == (60, 70)


################################################################
# False cases
  
def test_genomic_region_16():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 113, 118, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)
    #assert cds.induce(read) == False

def test_genomic_region_17():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 125, 131, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 145, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)
    #assert cds.induce(read) == False

def test_genomic_region_18():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 115, 125, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)
    #assert cds.induce(read) == False

def test_genomic_region_19():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 125, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 165, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)
    #assert cds.induce(read) == False

def test_genomic_region_20():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 164, 168, "+" ),
        HTSeq.GenomicInterval( "chr1", 175, 190, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)
    #assert cds.induce(read) == False

def test_genomic_region_21():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 60, 70, "+" ),
        HTSeq.GenomicInterval( "chr1", 80, 90, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)

def test_genomic_region_22():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 200, 210, "+" ),
        HTSeq.GenomicInterval( "chr1", 220, 230, "+" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds.induce(read)

################################################################
# Reverse strand

cds_ivs = [
    HTSeq.GenomicInterval( "chr1", 190, 200, "-" ),
    HTSeq.GenomicInterval( "chr1", 160, 180, "-" ),
    HTSeq.GenomicInterval( "chr1", 140, 150, "-" ),
    HTSeq.GenomicInterval( "chr1", 120, 130, "-" ),
    HTSeq.GenomicInterval( "chr1", 100, 110, "-" ),
]

cds_rev = GenomicRegion(cds_ivs)


def test_genomic_region_23():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 162, 175, "-" ),
        ]
    read = GenomicRegion(ivs)
    assert cds_rev.induce(read) == (15, 28)


def test_genomic_region_24():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 160, 175, "-" ),
        HTSeq.GenomicInterval( "chr1", 143, 150, "-" ),
        ]
    read = GenomicRegion(ivs)
    assert cds_rev.induce(read) == (15, 37)

def test_genomic_region_25():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 160, 175, "-" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "-"),
    ]
    read = GenomicRegion(ivs)
    assert cds_rev.induce(read) == (15, 40)

def test_genomic_region_26():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 160, 175, "-" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "-"),
        HTSeq.GenomicInterval( "chr1", 125, 130, "-"),
    ]
    read = GenomicRegion(ivs)
    assert cds_rev.induce(read) == (15, 45)

def test_genomic_region_27():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 195, 210, "-" ),
    ]
    read = GenomicRegion(ivs)
    #assert cds_rev.induce(read) == False
    assert cds_rev.induce(read) == (-10, 5)

################################################################
# False cases
def test_genomic_region_28():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 180, 210, "-" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds_rev.induce(read)
    #assert cds_rev.induce(read) == False


def test_genomic_region_29():
    ivs = [
        HTSeq.GenomicInterval( "chr1", 122, 126, "-" ),
        HTSeq.GenomicInterval( "chr1", 100, 115, "-" ),
    ]
    read = GenomicRegion(ivs)
    with pytest.raises(ValueError):
        cds_rev.induce(read)




################################################################
# test map()


cds_ivs = [
    HTSeq.GenomicInterval( "chr1", 100, 110, "+" ),
    HTSeq.GenomicInterval( "chr1", 120, 130, "+" ),
    HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
    HTSeq.GenomicInterval( "chr1", 160, 170, "+" ),
    HTSeq.GenomicInterval( "chr1", 180, 190, "+" ),
]

cds = GenomicRegion(cds_ivs)

def test_map_0():
    gr = cds.map((-10,-5))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 90, 95, "+" ),
        ])
    assert gr == r

def test_map_00():
    gr = cds.map((55,60))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 195, 200, "+" ),
        ])
    assert gr == r

def test_map_1():
    gr = cds.map((13,18))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 123, 128, "+" ),
        ])
    assert gr == r

def test_map_2():
    gr = cds.map((13,28))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 123, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 148, "+" ),
        ])
    assert gr == r

def test_map_3():
    gr = cds.map((13,37))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 123, 130, "+" ),
        HTSeq.GenomicInterval( "chr1", 140, 150, "+" ),
        HTSeq.GenomicInterval( "chr1", 160, 167, "+" ),
        ])
    assert gr == r

def test_map_4():
    gr = cds.map((10,20))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 120, 130, "+" ),
        ])
    assert gr == r

def test_map_5():
    gr = cds.map((-10,15))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 90, 110, "+" ),
        HTSeq.GenomicInterval( "chr1", 120, 125, "+" ),
        ])
    assert gr == r

def test_map_6():
    gr = cds.map((45,60))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 185, 200, "+" ),
        ])
    assert gr == r


################################################################
# test map() on reverse strand

cds_ivs = [
    HTSeq.GenomicInterval( "chr1", 190, 200, "-" ),
    HTSeq.GenomicInterval( "chr1", 160, 180, "-" ),
    HTSeq.GenomicInterval( "chr1", 140, 150, "-" ),
    HTSeq.GenomicInterval( "chr1", 120, 130, "-" ),
    HTSeq.GenomicInterval( "chr1", 100, 110, "-" ),
]

cds_rev = GenomicRegion(cds_ivs)

def test_map_7():
    gr = cds_rev.map((3,8))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 192, 197, "-" ),
        ])
    assert gr == r

def test_map_8():
    gr = cds_rev.map((3,18))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 190, 197, "-" ),
        HTSeq.GenomicInterval( "chr1", 172, 180, "-" ),
        ])
    assert gr == r

def test_map_9():
    gr = cds_rev.map((-10,5))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 195, 210, "-" ),
        ])
    assert gr == r

def test_map_10():
    gr = cds_rev.map((55,70))
    r = GenomicRegion([
        HTSeq.GenomicInterval( "chr1", 90, 105, "-" ),
        ])
    assert gr == r

