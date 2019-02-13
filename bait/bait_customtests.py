import os
import sys
import numpy as np
import logging


logger = logging.getLogger(__name__)


# --------------------------------------------- Private


def _normalizeTrace(workList, rangeVal=[-1, 1]):
    """
    This simple method will normalize the trace between rangeVal.
    Simply by scaling everything...

    """
    minVal, maxVal = min(workList), max(workList)
    workList[:] = [((x - minVal) / (maxVal - minVal)) * (rangeVal[1] - rangeVal[0]) for x in workList]
    workList = workList + rangeVal[0]
    return workList


def _createCF(inarray):
    """
    Simple method to create the carachteristic function of BaIt
    picking algorithm

    """
    # ORIGINAL: outarray = abs(inarray)
    outarray = abs(inarray**2)      # MB 13.02.2019
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
    PostPick_GMT = bpd['pickUTC']+timewin
    wt.data = _createCF(wt.data)
    wt.slice(bpd['pickUTC'], PostPick_GMT)
    # ------ Out + Log
    if wt.data.max() <= thr_par_1:
        logger.debug((' '*4+'FALSE  %s: %5.3f > %5.3f'+os.linesep) % (
                                            tfn, wt.data.max(), thr_par_1))
        return (False, wt.data.max())
    else:
        logger.debug((' '*4+'TRUE   %s: %5.3f > %5.3f'+os.linesep) % (
                                            tfn, wt.data.max(), thr_par_1))
        return (True, wt.data.max())


def SignalSustain(wt, bpd, timewin, timenum, snratio):
    """
    This test evaluate the mean value of signal windows in comparison
    with the noise window before the pick. The ratio should be
    higher than a threshold given by user.

        INPUT:
            - workTrace (obspy.Trace obj)
            - iteration number [to reach the proper pick]
            - config py file (BaIt_Config)
            - File ID (already opened) for logs
        OUTPUT
            - bool (True/False)

    """
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
    # ------ Out + Log
    if True in (window.mean() <= snratio * Noise.data.mean()
                for window in WINDOWING):
        # signaal2noise ratio is lower then user-threshold
        logger.debug((' '*4+'FALSE  %s: [SNratio] %5.3f >' +
                      ' [SigMean(w)/NoiseMean(w)] %5.3f') %
                     (tfn, snratio, [float(_xx.mean() / Noise.data.mean())
                      for _xx in WINDOWING]))
        return (False, [float(_xx.mean() / Noise.data.mean())
                for _xx in WINDOWING])
    else:
        logger.debug((' '*4+'TRUE   %s: [SNratio] %5.3f <' +
                      ' [SigMean(w)/NoiseMean(w)] %5.3f') %
                     (tfn, snratio, [float(_xx.mean() / Noise.data.mean())
                      for _xx in WINDOWING]))
        return (True, [float(_xx.mean() / Noise.data.mean())
                for _xx in WINDOWING])


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
