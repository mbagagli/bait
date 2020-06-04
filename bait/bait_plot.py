import sys
from obspy.core.trace import Trace
import logging
import numpy as np
# plot
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

logger = logging.getLogger(__name__)


# ---------------------------


def plotBait(tr,
             cf,
             baitdict,
             figtitle=None,
             show=False,
             savefig=False,
             savepath=None):
    """
    Improved method to plot all the picks
    """
    tfn = sys._getframe().f_code.co_name
    if not isinstance(tr, Trace):
        logger.error('%s: not a valid ObsPy trace (TR) ...' % tfn)
        return False
    if not isinstance(cf, Trace):
        logger.error('%s: not a valid ObsPy trace (CF) ...' % tfn)
        return False
    #
    fig = plt.figure(figsize=(8, 4.5))

    ax1 = plt.subplot(211)
    ax1.plot(tr.times("matplotlib"), tr.data, color='black')

    ax2 = plt.subplot(212, sharex=ax1)  # , sharey=ax1)
    ax2.plot(cf.times("matplotlib"), cf.data, color='blue')

    # ---- Plot picks
    for _kk, _vv in baitdict.items():
        if _vv['pickUTC']:
            if _vv['evaluatePick']:
                tmpcol = '#008081'
                tmplab = 'BK'
                tmplst = 'solid'
            else:
                tmpcol = 'black'
                tmplab = 'picks'
                tmplst = 'dashed'
            #
            for _ax in (ax1, ax2):
                # _ax.axvline(date2num(_vv['pickUTC'].datetime),
                _ax.axvline(
                    _vv['pickUTC'].datetime,
                    color=tmpcol, linewidth=2, linestyle=tmplst, label=tmplab)
                if _vv['pickUTC_AIC']:
                    _ax.axvline(_vv['pickUTC_AIC'].datetime,
                                color='gold',
                                linewidth=2,
                                linestyle='solid',
                                label='AIC')
    # ---- Finalize

    # fig
    if figtitle:
        ax1.set_title(figtitle, fontsize=15, fontweight='bold')
    fig.set_tight_layout(True)

    # ax1
    handles, labels = ax1.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='lower left')
    # ax1.xaxis.set_major_formatter(AutoDateFormatter('%H:%M:%S'))

    # ax2
    handles, labels = ax2.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), loc='upper left')
    # ax2.xaxis.set_major_formatter(AutoDateFormatter('%H:%M:%S'))

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    # ---- Export / Show
    if savefig:
        if not savepath:
            savepath = "baitfig.pdf"
        fig.savefig(savepath, bbox_inches='tight')
    if show:
        plt.show()
    return fig, (ax1, ax2)


def plotAmplitudeTest(tr, cf, baitdict, idx, timewin, thr_1, show=False):
    """ Plotting routine for the Amplitude Test over a defined pick """
    tfn = sys._getframe().f_code.co_name
    if not isinstance(tr, Trace):
        logger.error('%s: not a valid ObsPy trace (TR) ...' % tfn)
        return False
    if not isinstance(cf, Trace):
        logger.error('%s: not a valid ObsPy trace (CF) ...' % tfn)
        return False
    #
    fig = plt.figure(figsize=(8, 4.5))

    ax1 = plt.subplot(211)
    ax1.plot(tr.times("matplotlib"), tr.data, color='black')

    ax2 = plt.subplot(212, sharex=ax1)  # , sharey=ax1)
    ax2.plot(cf.times("matplotlib"), cf.data, color='blue')

    _vv = baitdict[str(idx)]
    testpass = _vv['evaluatePick_tests']['SignalAmp'][0]
    maxres = _vv['evaluatePick_tests']['SignalAmp'][1]

    if _vv['pickUTC']:
        if _vv['evaluatePick']:
            tmpcol = '#008081'
            tmplab = 'BK'
            tmplst = 'solid'
        else:
            tmpcol = 'black'
            tmplab = 'picks'
            tmplst = 'dashed'
        #
        for _ax in (ax1, ax2):
            _ax.axvline(
                _vv['pickUTC'].datetime,
                color=tmpcol, linewidth=2, linestyle=tmplst, label=tmplab)
            _ax.set_xlim((_vv['pickUTC']-(2*timewin)).datetime,
                         (_vv['pickUTC']+(2*timewin)).datetime)

    # ---------- Adding Test results
    ax1.axvline((_vv['pickUTC'] + timewin).datetime, color='darkgray', linewidth=1.5, linestyle='solid')
    ax2.axvline((_vv['pickUTC'] + timewin).datetime, color='darkgray', linewidth=1.5, linestyle='solid')

    if testpass:
        ss = _vv['pickUTC'].datetime
        se = (_vv['pickUTC'] + timewin).datetime
        ax2.plot((ss, se),
                 (maxres, maxres),
                 color='forestgreen', label=('Max: %4.2f' % maxres))
    else:
        ss = _vv['pickUTC'].datetime
        se = (_vv['pickUTC'] + timewin).datetime
        ax2.plot((ss, se),
                 (maxres, maxres),
                 color='red', label=('Max: %4.2f' % maxres))
    #
    fig.suptitle("Signal Amp - %s: %4.2f" % (testpass, maxres),
                 fontweight="bold", fontsize=15)

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    if show:
        plt.show()
    return fig, (ax1, ax2)


def plotSustainTest(tr, cf, baitdict, idx, timewin, timenum, snratio, mode="mean",
                    failwindow_tolerance=0, show=False):
    """ Plotting routine for the Sustain Test over a defined pick """
    tfn = sys._getframe().f_code.co_name
    if not isinstance(tr, Trace):
        logger.error('%s: not a valid ObsPy trace (TR) ...' % tfn)
        return False
    if not isinstance(cf, Trace):
        logger.error('%s: not a valid ObsPy trace (CF) ...' % tfn)
        return False
    #
    fig = plt.figure(figsize=(8, 4.5))

    ax1 = plt.subplot(211)
    ax1.plot(tr.times("matplotlib"), tr.data, color='black')

    ax2 = plt.subplot(212, sharex=ax1)  # , sharey=ax1)
    ax2.plot(cf.times("matplotlib"), cf.data, color='blue')

    _vv = baitdict[str(idx)]
    testpass = _vv['evaluatePick_tests']['SignalSustain'][0]
    ratios = _vv['evaluatePick_tests']['SignalSustain'][1]

    if _vv['pickUTC']:
        if _vv['evaluatePick']:
            tmpcol = '#008081'
            tmplab = 'BK'
            tmplst = 'solid'
        else:
            tmpcol = 'black'
            tmplab = 'picks'
            tmplst = 'dashed'
        #
        for _ax in (ax1, ax2):
            # _ax.axvline(date2num(_vv['pickUTC'].datetime),
            _ax.axvline(
                _vv['pickUTC'].datetime,
                color=tmpcol, linewidth=2, linestyle=tmplst, label=tmplab)
            _ax.set_xlim((_vv['pickUTC']-(2*timewin)).datetime,
                         (_vv['pickUTC']+((timenum+1)*timewin)).datetime)

    # ---------- Adding Test results
    _noi_mean = cf.slice(_vv['pickUTC'] - timewin, _vv['pickUTC']).data.mean()
    # _noi_max = cf.slice(_vv['pickUTC'] - timewin, _vv['pickUTC']).data.mean()
    ax1.axvline((_vv['pickUTC'] - timewin).datetime, color='darkgray',
                linewidth=1.5, linestyle='solid')
    ax2.axvline((_vv['pickUTC'] - timewin).datetime, color='darkgray',
                linewidth=1.5, linestyle='solid')
    ax2.plot(((_vv['pickUTC'] - timewin).datetime, _vv['pickUTC'].datetime),
             (_noi_mean, _noi_mean), color="gold")

    for _xx in range(1, timenum+1):
        ax2.axvline((_vv['pickUTC'] + (timewin*_xx)).datetime, color='darkgray',
                    linewidth=1.5, linestyle='solid')
        lutc = _vv['pickUTC'] + ((_xx-1) * timewin)
        uutc = _vv['pickUTC'] + (_xx*timewin)
        if mode.lower() == "mean":
            _sig_data = cf.slice(lutc, uutc).data.mean()
        else:
            _sig_data = cf.slice(lutc, uutc).data.max()
        #
        if ratios[(_xx-1)] <= snratio:
            ax2.plot((lutc.datetime, uutc.datetime),
                     (_sig_data, _sig_data),
                     color='red', linewidth=1.5)
        else:
            ax2.plot((lutc.datetime, uutc.datetime),
                     (_sig_data, _sig_data),
                     color='forestgreen', linewidth=1.5)

    fig.suptitle("Signal Sustain - %s %s" % (mode.upper(), testpass),
                 fontweight="bold", fontsize=15)

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    if show:
        plt.show()
    return fig, (ax1, ax2)


def plotLowFreqTest(tr, cf, baitdict, idx, timewin, conf, show=False):
    """ Plotting routine for the Sustain Test over a defined pick """
    tfn = sys._getframe().f_code.co_name
    if not isinstance(tr, Trace):
        logger.error('%s: not a valid ObsPy trace (TR) ...' % tfn)
        return False
    if not isinstance(cf, Trace):
        logger.error('%s: not a valid ObsPy trace (CF) ...' % tfn)
        return False
    #
    fig = plt.figure(figsize=(8, 4.5))

    ax1 = plt.subplot(211)
    ax1.plot(tr.times("matplotlib"), tr.data, color='black')

    ax2 = plt.subplot(212, sharex=ax1)  # , sharey=ax1)
    ax2.plot(cf.times("matplotlib"), cf.data, color='blue')

    _vv = baitdict[str(idx)]
    testpass = _vv['evaluatePick_tests']['LowFreqTrend'][0]
    percent = _vv['evaluatePick_tests']['LowFreqTrend'][1]

    if _vv['pickUTC']:
        if _vv['evaluatePick']:
            tmpcol = '#008081'
            tmplab = 'BK'
            tmplst = 'solid'
        else:
            tmpcol = 'black'
            tmplab = 'picks'
            tmplst = 'dashed'
        #
        for _ax in (ax1, ax2):
            _ax.axvline(
                _vv['pickUTC'].datetime,
                color=tmpcol, linewidth=2, linestyle=tmplst, label=tmplab)
            _ax.set_xlim((_vv['pickUTC']-(2*timewin)).datetime,
                         (_vv['pickUTC']+(2*timewin)).datetime)

    # ---------- Adding Test results
    ax1.axvline((_vv['pickUTC'] + timewin).datetime, color='darkgray',
                linewidth=1.5, linestyle='solid')
    ax2.axvline((_vv['pickUTC'] + timewin).datetime, color='darkgray',
                linewidth=1.5, linestyle='solid')

    _tmpCF = cf.slice(_vv['pickUTC'], _vv['pickUTC'] + timewin)
    _tmpCF_times = _tmpCF.times("matplotlib")
    _tmpCF_data = _tmpCF.data
    diffcf = np.diff(_tmpCF.data)

    for _xx, _vv in enumerate(diffcf):
        if _vv >= 0:
            # positive trend
            ax2.plot((_tmpCF_times[_xx], _tmpCF_times[_xx+1]),
                     (_tmpCF_data[_xx], _tmpCF_data[_xx+1]),
                     color='forestgreen', label="Positive Sign", lw=2)
        else:
            # negative trend
            ax2.plot((_tmpCF_times[_xx], _tmpCF_times[_xx+1]),
                     (_tmpCF_data[_xx], _tmpCF_data[_xx+1]),
                     color='red', label="Negative Sign", lw=2)

    fig.suptitle("Low Freq Trend - %s: %4.2f (+) %4.2f (-) TOLERANCE: %4.2f" %
                 (testpass, percent[0], percent[1], conf),
                 fontweight="bold", fontsize=15)

    # NEW: the next call should take care of the X-axis formatting on zoom
    AutoDateFormatter(AutoDateLocator())

    if show:
        plt.show()
    return fig, (ax1, ax2)
