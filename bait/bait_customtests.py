"""
DEVELOPER HINT:
 - Every main method should receive a trace.copy() instance
 - If you want to use SLICE instead of TRIM, make sure that your
   pointer reference changes as well! (i.e. reassign a new name to the slice)
        ES: >>> Signal = wt.slice( ...)
                    NOT
            >>>  wt.slice(...)


"""

import sys
import numpy as np
import logging
from bait import bait_errors as BE


logger = logging.getLogger(__name__)


# --------------------------------------------- Private


def _normalizeTrace(workList, rangeVal=[-1, 1]):
    """
    This simple method will normalize the trace between rangeVal.
    Simply by scaling everything...

    """
    minVal, maxVal = min(workList), max(workList)
    workList[:] = [((x - minVal) / (maxVal - minVal)) *
                   (rangeVal[1] - rangeVal[0]) for x in workList]
    workList = workList + rangeVal[0]
    return workList


def _createCF(inarray):
    """
    Simple method to create the carachteristic function of BaIt
    picking algorithm

    *** NB: The outarray of 13.02.2019 (the squared one), better enanche
            impulsive features of the signal, but it's really weak on
            emergent arrivals, especially with The SignalAmp feature.
    """
    outarray = np.abs(inarray)         # ORIGINAL
    # outarray = abs(inarray**2)      # MB 13.02.2019 - test -
    # outarray = np.sqrt(abs(inarray))      # MB 13.02.2019 - test -
    outarray = _normalizeTrace(outarray, rangeVal=[0, 1])
    return outarray


# --------------------------------------------- Evaluation


def SignalAmp(wt, bpd, timewin, thr_par_1):
    """
    This test evaluate the maximum amplitude of the first window
    after the pick and compare it to a threshold given by user.
    max amp after [pick must be higher than threshold

        INPUT:
            - workTrace (obspy.Trace obj)
            - bpd = baitpickdict with the actual pick info to analyze

        OUTPUT
            - bool (True/False)

    """
    tfn = sys._getframe().f_code.co_name
    wt.data = _createCF(wt.data)
    Signal = wt.slice(bpd['pickUTC'], bpd['pickUTC'] + timewin)

    # ------ Out + Log
    if Signal.data.max() <= thr_par_1:
        logger.debug((' '*4+'FALSE  %s: %5.3f < %5.3f') % (
                                            tfn, Signal.data.max(), thr_par_1))
        return (False, Signal.data.max())
    else:
        logger.debug((' '*4+'TRUE   %s: %5.3f > %5.3f') % (
                                            tfn, Signal.data.max(), thr_par_1))
        return (True, Signal.data.max())


def SignalSustain(wt, bpd, timewin, timenum, snratio, mode="mean",
                  failwindow_tolerance=0):
    """
    This test evaluate the mean value of signal windows in comparison
    with the noise window before the pick. The ratio should be
    higher than a threshold given by user.

        INPUT:
            - wt: workTrace (obspy.Trace obj)
            - bpd: BaIt object
            - timewin: amount of time for each windows (also the one for noise)
            - timenum: number of windows AFTER the pick
            - snratio: signal/noise ratio threshold. To pass must be HIGHER
            - mode ["mean"]: compare the CF SIGNAL MEAN against CF NOISE MEAN
                   ["max"]: compare the CF SIGNAL MAX against CF NOISE MEAN
            - failwindow_tolerance: number of windows that could be
                                    BELOW threshold ratio.
                                    NB: the FIRST on must be ALWAYS up
                                    (because is what BK see in triggering)

        OUTPUT
            - tuple: Result (bool), Values (snr each windows)

    """
    if failwindow_tolerance > timenum:
        failwindow_tolerance = timenum

    tfn = sys._getframe().f_code.co_name
    PrePick_GMT = bpd['pickUTC']-timewin
    wt.data = _createCF(wt.data)
    #
    Noise = wt.slice(PrePick_GMT, bpd['pickUTC'])
    #
    WINDOWING = []                # list of numpy array
    for num in range(timenum):
        # index must start from 0
        Signal = wt.slice(bpd['pickUTC'] + (num*timewin),
                          bpd['pickUTC'] + ((num+1)*timewin))
        WINDOWING.append(Signal.data)

    # Next steps cannot be included in list-comprehension because of
    # Exception handling
    RATIOS = []
    if mode.lower() == "mean":
        for _xx in WINDOWING:
            try:
                RATIOS.append(float(_xx.mean() / Noise.data.mean()))
            except ValueError:
                RATIOS.append(np.nan)

    elif mode.lower() == "max":
        # Because numpy.arrayy max e min raise exception if array empty!
        for _xx in WINDOWING:
            try:
                RATIOS.append(float(_xx.max() / Noise.data.mean()))
            except ValueError:
                RATIOS.append(np.nan)
    else:
        raise BE.InvalidParameter("MODE parameter must be either MAX or MEAN!")

    _boolarr = np.array([True if _xx <= snratio else False for _xx in RATIOS])
    if np.sum(_boolarr) <= failwindow_tolerance:
        # True if below the threshold -> 1st WINDOW must be ALWAYS pass
        if _boolarr[0]:
            PASS = False
        else:
            PASS = True
    else:
        PASS = False

    # --- Log
    if PASS:
        logger.debug((' '*4+'TRUE   %s: [SNratio] %5.3f <' +
                      ' [Ratios] %s [Tolerance: %d]') %
                     (tfn, snratio, RATIOS, failwindow_tolerance))
        return (True, RATIOS)
        # return (True, (RATIOS, np.sum(_boolarr)))
    else:
        logger.debug((' '*4+'FALSE   %s: [SNratio] %5.3f <' +
                      ' [Ratios] %s [Tolerance: %d]') %
                     (tfn, snratio, RATIOS, failwindow_tolerance))
        return (False, RATIOS)
        # return ( False, (RATIOS, np.sum(_boolarr)))


def LowFreqTrend(wt, bpd, timewin, conf=0.95):
    """
    This method should help avoiding mispicks due
    to the so-called filter effect by recognizing trends (pos or negative)
    return False if trend found --> bad pick

    """
    tfn = sys._getframe().f_code.co_name
    # ------ WORK
    # wt.data = _createCF(wt.data)
    wt.slice(bpd['pickUTC'], bpd['pickUTC'] + timewin)

    # asign=np.sign(wt.data)
    asign = np.sign(np.diff(wt.data))
    unique, counts = np.unique(asign, return_counts=True)
    dsign = dict(zip(unique, counts))
    #
    for key in (-1.0, 1.0):
        if key in dsign and dsign[key]:
            pass
        else:
            dsign[key] = 0

    # ------ Out + Log
    if dsign[1.0]/len(asign) >= conf or dsign[-1.0]/len(asign) >= conf:
        logger.debug((' '*4+'FALSE  %s: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]') %
                     (tfn, dsign[1.0]/len(asign), dsign[-1.0]/len(asign), conf))
        return (False, (dsign[1.0]/len(asign), dsign[-1.0]/len(asign), conf))
    else:
        logger.debug((' '*4+'TRUE   %s: Pos. %5.2f  -  Neg. %5.2f  [%5.2f]') %
                     (tfn, dsign[1.0]/len(asign), dsign[-1.0]/len(asign), conf))
        return (True, (dsign[1.0]/len(asign), dsign[-1.0]/len(asign), conf))


# --------------------------------------------- Phase recognition
# TIPS
# this_function_name = sys._getframe().f_code.co_name
