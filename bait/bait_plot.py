import sys
from obspy.core.trace import Trace
import logging
from bait import bait_customtests as BCT
# plot
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib.dates import AutoDateLocator, AutoDateFormatter

logger = logging.getLogger(__name__)


# ---------------------------


def plotBait(tr,
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
        logger.error('%s: not a valid ObsPy trace ...' % tfn)
        return False
    #
    fig = plt.figure(figsize=(8,4.5))

    CF_Trace = tr.copy()
    CF_Trace.data = BCT._createCF(CF_Trace.data)

    ax1 = plt.subplot(211)
    ax1.plot(tr.times("matplotlib"), tr.data, color='black')

    ax2 = plt.subplot(212, sharex=ax1)  # , sharey=ax1)
    ax2.plot(CF_Trace.times("matplotlib"), CF_Trace.data, color='blue')

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
                _ax.axvline(_vv['pickUTC'].datetime,
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
        fig.savefig(savepath, bbox_inches='tight', dpi=310)
    if show:
        plt.show()
    return fig, (ax1, ax2)
