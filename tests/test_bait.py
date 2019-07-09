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
          'LowFreqTrend': [0.2, 0.80],
    },
    'pickAIC': True,
    'pickAIC_conf': {
          'useraw': True,
          'wintrim_noise': 0.8,
          'wintrim_sign': 0.5,
    }
}

# --------------------


straw = read()
stproc = miniproc(straw)


def test_setworktrace():
    """ Test the picker """
    errors = []
    BP = BaIt(stproc, stream_raw=straw, channel="*Z")
    BP._setworktrace("*Z", "PROC")
    pr = BP.wt
    BP._setworktrace("*Z", "RAW")
    ra = BP.wt
    #
    if np.array_equal(pr.data, ra.data):
        errors.append("Returned picks are equal, no difference")
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_raisingerror():
    errors = []
    #
    BP = BaIt(stproc,
              stream_raw=straw,
              channel="*Z",
              **BAIT_PAR_DICT)
    #
    BP.pickAIC = False
    BP.pickAIC_conf = {}
    #
    BP.CatchEmAll()

    # ========================================== Tests
    raised = False
    try:
        BP.getTruePick(idx=0, picker="AIC", compact_format=False)
    except BE.MissingAttribute:
        raised = True
    #
    if not raised:
        errors.append("AIC selection error not raised!")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_returnedpick_single():
    errors = []
    #
    BP = BaIt(stproc,
              stream_raw=straw,
              channel="*Z",
              **BAIT_PAR_DICT)
    #
    BP.CatchEmAll()
    #
    # BP.plotPicks(show=True)
    # ========================================== Tests

    # print("%r" % BP.getTruePick(idx=0, picker="AIC", compact_format=True)[0])
    if (BP.getTruePick(idx=0, picker="AIC", compact_format=True)[0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 7, 750000)):
        errors.append("P1 AIC not correct")

    # print("%r" % BP.getTruePick(idx=0, picker="BK", compact_format=True)[0])
    if (BP.getTruePick(idx=0, picker="BK", compact_format=True)[0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 7, 720000)):
        errors.append("P1 BK not correct")

    # print("%r" % BP.getTruePick(idx=1, picker="AIC", compact_format=True)[0])
    if (BP.getTruePick(idx=1, picker="AIC", compact_format=True)[0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 8, 710000)):
        errors.append("P2 AIC not correct")

    # print("%r" % BP.getTruePick(idx=1, picker="BK", compact_format=True)[0])
    if (BP.getTruePick(idx=1, picker="BK", compact_format=True)[0] !=
       UTCDateTime(2009, 8, 24, 0, 20, 8, 740000)):
        errors.append("P2 BK not correct")
    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))


def test_returnedpick_all():
    errors = []
    #
    BP = BaIt(stproc,
              stream_raw=straw,
              channel="*Z",
              **BAIT_PAR_DICT)
    #
    BP.CatchEmAll()
    #
    # BP.plotPicks(show=True)
    # ========================================== Tests

    picklist = BP.getTruePick(idx="all", picker="AIC", compact_format=True)

    if len(picklist) != 2:
        errors.append("Wrong AIC list length returned")

    if not picklist[0][0] < picklist[1][0]:
        errors.append("Pick list AIC not sorted in UTCtime!")

    if picklist[0][0] != UTCDateTime(2009, 8, 24, 0, 20, 7, 750000):
        errors.append("P1 AIC not correct")

    if picklist[1][0] != UTCDateTime(2009, 8, 24, 0, 20, 8, 710000):
        errors.append("P2 AIC not correct")

    # ------

    picklist = BP.getTruePick(idx="all", picker="BK", compact_format=True)

    if len(picklist) != 2:
        errors.append("Wrong BK list length returned")

    if not picklist[0][0] < picklist[1][0]:
        errors.append("Pick list BK not sorted in UTCtime!")

    if picklist[0][0] != UTCDateTime(2009, 8, 24, 0, 20, 7, 720000):
        errors.append("P1 BK not correct")

    if picklist[1][0] != UTCDateTime(2009, 8, 24, 0, 20, 8, 740000):
        errors.append("P2 BK not correct")

    # ------

    picklist = BP.getTruePick(idx="all", picker="AIC", compact_format=False)

    if len(picklist) != 2:
        errors.append("Wrong ALL AIC pick list length returned")

    if not picklist[0]['iteration'] < picklist[1]['iteration']:
        errors.append("Iteration ALL AIC sorting not respected")

    # ------

    picklist = BP.getTruePick(idx="all", picker="BK", compact_format=False)

    if len(picklist) != 2:
        errors.append("Wrong ALL BK pick list length returned")

    if not picklist[0]['iteration'] < picklist[1]['iteration']:
        errors.append("Iteration ALL BK sorting not respected")

    #
    assert not errors, "Errors occured:\n{}".format("\n".join(errors))
