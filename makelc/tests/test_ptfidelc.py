from makelc.ptfidelc import ptfide_light_curve

def test_ptfide_light_curve():
    """Read in PTFIDE output for 16cbx to test ptfide_light_curve"""
    lc16cbx = ptfide_light_curve("../data/forcepsffitdiff_d4636_f1_c8.out", 
                                 hjd0 = 2457250, SNT = 4)
    assert len(lc16cbx[0]) == 6
    assert len(lc16cbx[3]) == 111
    assert len(lc16cbx[6]) == 117