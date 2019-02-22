from bait.bait import BaIt
from obspy import read
import numpy as np


def miniproc(st):
    prs = st.copy()
    prs.detrend('demean')
    prs.detrend('simple')
    prs.taper(max_percentage=0.05, type='cosine')
    prs.filter("bandpass",
               freqmin=1,
               freqmax=30,
               corners=2,
               zerophase=True)
    return prs

# --------------------

straw = read()
stproc = miniproc(straw)

def test_setworktrace():
    """ Test the picker """
    errors = []
    BP = BaIt(stproc, stream_raw=straw, channel="*Z")
    BP._setworktrace("*Z", "PROC");
    pr = BP.wt
    BP._setworktrace("*Z", "RAW")
    ra = BP.wt
    #
    if np.array_equal(pr.data, ra.data):
        errors.append("Returned picks are equal, no difference")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
