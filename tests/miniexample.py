
from bait.bait import BaIt
import bait.bait_errors as BE
from obspy import read, UTCDateTime
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


BAIT_PAR_DICT = {
    'max_iter': 10,
    'opbk_main': {
          'tdownmax': 0.1,     # float: seconds depends on filtering
          'tupevent': 0.5,     # float: seconds depends on filtering
          'thr1': 6.0,         # float: sample for CF's value threshold
          'thr2': 10.0,        # float: sample for sigma updating threshold
          'preset_len': 0.6,   # float: seconds
          'p_dur': 1.0         # float: seconds
    },
    'opbk_aux': {
          'tdownmax': 0.1,
          'tupevent': 0.25,    # time [s] for CF to remain above threshold γ
          'thr1': 3,           # 10 orig
          'thr2': 6,           # 20 orig
          'preset_len': 0.1,   # sec
          'p_dur': 1.0         # sec
    },
    'test_pickvalidation': {
          'SignalAmp': [0.5, 0.05],
          'SignalSustain': [0.2, 5, 1.2],
          'LowFreqTrend': [0.2, 0.80]
    },
    'pickAIC': True,
    'pickAIC_conf': {
          'useraw': True,
          'wintrim_noise': 0.8,
          'wintrim_sign': 0.5
    }
}


straw = read("../tests_data/obspyread.mseed")
stproc = miniproc(straw)

# ========================================== Init
BP = BaIt(stproc,
          stream_raw=straw,
          channel="*Z",
          **BAIT_PAR_DICT)

# ========================================== Picks
BP.CatchEmAll()
picklist = BP.extract_true_pick(idx="ALL",
                                picker=["AIC", "BK"],
                                compact_format=True)
print("Valid picks found:")
for xx, ii in enumerate(picklist):
    print("%d -  %s  @  %s" % (xx+1, ii[1], ii[0]))

# ========================================== Plots
BP.plotPicks(show=True)
BP.plotTests(1, name="all", plotraw=False)
BP.plotTests(2, name="all", plotraw=False)
