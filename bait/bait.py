"""
This module is extracted from BaIt v2.1.0.
This object-oriented iterative picking algorithm is hopefully
performing better than the previous sequential one.
"""

# lib for MAIN
import logging
# lib for BAIT
from bait import bait_errors as BE
from bait import bait_plot as BP
from bait import bait_customtests as BCT
from obspy.signal.trigger import pk_baer
from operator import itemgetter
import numpy as np
# lib for Errors
from obspy.core.trace import Trace
from obspy.core.stream import Stream

logger = logging.getLogger(__name__)


# ------------------------------------------------------ BAIT

class BaIt(object):
    """
    This module represent the standalone version of BAIT picking algorithm
    readjusted for quake.

    The algorithm will pick consecutively different waveform's protions
    by trimming the data from previous pick to tihe end-of-trace.
    If pick is found, evaluation is pursued, otherwise returns
    no valid picks.

    *** NB If stream_raw==None even if user AIC config specify the
    raw option, the processed will be used instead, without throwing
    any errors

    """
    def __init__(self,
                 stream,
                 stream_raw=None,
                 channel="*Z",
                 max_iter=5,
                 opbk_main={},
                 opbk_aux={},
                 test_pickvalidation={},
                 test_postvalidation={},
                 pickAIC=None,
                 pickAIC_conf={}):
        self.st = stream
        self.straw = stream_raw
        self.wc = channel
        self.wt = None                          # workingtrace
        self._setworktrace(channel, "PROC")
        self.maxit = max_iter
        self.opbk_main = opbk_main
        self.opbk_aux = opbk_aux
        #
        self.pick_test = test_pickvalidation
        self.post_test = test_postvalidation
        self.baitdict = {}
        self.baitdict_keys = ('pickUTC',
                              'bk_info',
                              'pickUTC_AIC',
                              'AICcf',
                              'iteration',
                              'evaluatePick',
                              'evaluatePick_tests')
        self.baitdict_post = {}
        self.baitdict_post_keys = ('validPicks',
                                   'evaluatePickPost_tests')
        self._createbaitdictkey_POST()
        # AiC
        self.pickAIC = pickAIC
        self.pickAIC_conf = pickAIC_conf

    def _sec2sample(self, value, df):
        """
        Utility method to define convert USER input parameter (seconds)
        into obspy pickers 'n_sample' units.

        Python3 round(float)==int // Python2 round(float)==float
        BETTER USE: int(round(... to have compatibility

        Formula: int(round( INSEC * df))
        *** NB: input sec could be float
        """
        return int(round(value * df))

    def _createbaitdictkey(self, addkey):
        """ Method to take care about the creation of new dict keys """
        # self.baitdict[str(addkey)] = {_kk: None for _kk in self.baitdict_keys}
        self.baitdict[str(addkey)] = {_kk: (None if _kk not in (
                                                    'evaluatePick_tests'
                                     ) else {}) for _kk in self.baitdict_keys}
        #
        if self.pick_test:
            for _kk in self.pick_test.keys():
                self.baitdict[str(addkey)]["evaluatePick_tests"][_kk] = None

    def _createbaitdictkey_POST(self):
        """ Store the final, complete evaluation step in class """
        if self.post_test:
            for _kk in self.post_test.keys():
                self.baitdict_post["evaluatePickPost_tests"][_kk] = None

    def _storepick(self, stkey, **kwargs):
        """
        This is the method that is called FIRST by bait algorithm.
        Is called by `picker` method.

        *** This method take care to initialize missing iteraion
            dict keys in self.baitdict

        """
        if str(stkey) not in self.baitdict.keys():
            self._createbaitdictkey(str(stkey))
        #
        for _kk, _vv in kwargs.items():
            if _kk not in self.baitdict_keys:
                raise BE.MissingKey()
            self.baitdict[str(stkey)][_kk] = _vv

    def _setworktrace(self, channel, procraw):
        """
        Private method to change pointer of working trace for picker.
        Use obspy.Stream.select function
        """
        if procraw.lower() not in ('raw', 'proc'):
            self.wt = None
            self.wc = None
            raise BE.BadKeyValue({'message': ("wrong tag selection --> %s" %
                                              procraw)})
        #
        if procraw.lower() == "raw" and isinstance(self.straw, Stream):
            selstream = self.straw
        else:
            selstream = self.st
        #
        self.wt = selstream.select(channel=channel)[0]
        self.wc = channel

    def _getbaitdict(self):
        return self.baitdict

    def CatchEmAll(self):
        """
        Main algorithm that calls the picker and tests.
        Internally it stores the results as well.
        """
        # It breaks until BP obspy cannot pick anymore, otherwise goes till MAXIT!
        # *** nextline is a switch --> if at least one pick is accepted,
        #                              is changed to True
        VALIDPICKS = False
        for ITERATION in range(1, self.maxit + 1):
            # The pick (if present), it stored by the picker function
            #
            if ITERATION == 1:
                self.picker(ITERATION, **self.opbk_main)
            else:
                self.picker(ITERATION, **self.opbk_aux)
            #
            if self.baitdict[str(ITERATION)]['pickUTC']:
                # pick found -> evaluate
                if self.pick_test:
                    # Check if PICK_BK is valid
                    self.baitdict[str(ITERATION)]['evaluatePick'] = (
                        self.evaluatePick_BK(str(ITERATION))
                    )
                    logger.debug("PickAccepted: %s - Results: %s" % (
                        self.baitdict[str(ITERATION)]['evaluatePick'],
                        self.baitdict[str(ITERATION)]['evaluatePick_tests']))

                if self.baitdict[str(ITERATION)]['evaluatePick']:
                    VALIDPICKS = True
                    # If pick is valid and user wants AIC --> call AIC picker
                    if self.pickAIC:
                        aicpick, aicfun, aicidx = self.AIC(
                          aroundpick=self.baitdict[str(ITERATION)]['pickUTC'],
                          **self.pickAIC_conf)
                        self._storepick(ITERATION,
                                        pickUTC_AIC=aicpick,
                                        AICcf=aicfun)

            else:     # no pick -> exit the loop
                logger.error(('~~~ No pick @ Iteration %d') % (ITERATION))
                break
        # Post - Picking
        if not VALIDPICKS:
            logger.error("*** No valid pick found")
            raise BE.MissingVariable({'message': "No TRUE pick found!"})
        else:
            if self.post_test:
                self.evaluatePick_BK_POST()

    def picker(self, it, tdownmax, tupevent, thr1, thr2, preset_len, p_dur):
        """
        This method is a wrapper to the autopicker Baer&Kradofler
        The dict keys are 'None' type if no pick found.

        RETURN UTCDateTime pick and Pick INFO from OBsPy BK

        """
        # --- Select trace 22022019 --> v2.1.6
        self._setworktrace(self.wc, "PROC")  # baer picker needs always proc
        if not isinstance(self.wt, Trace):
            raise BE.BadInstance()
        # -------------------------------------------------------- Cut Trace
        tr = self.wt.copy()
        if it > 1:
            tr = tr.trim(self.baitdict[str(it-1)]["pickUTC"],
                         self.wt.stats.endtime)

        # ------------------------------------------------- v.1.1 sample2sec
        # mainly we change the input from BaIt_Config in SECONDS and convert here in SAMPLES
        # Python3 round(float)==int // Python2 round(float)==float --> int(round(... to have compatibility
        df = tr.stats.sampling_rate
        preset_len_NEW = self._sec2sample(preset_len, df)
        tupevent_NEW = self._sec2sample(tupevent, df)            # tupevent: should be the inverse of high-pass
                                                                 #           freq or low freq in bandpass
        tdownmax_NEW = self._sec2sample(tdownmax, df)            # Half of tupevent
        p_dur_NEW = self._sec2sample(p_dur, df)                  # time-interval in which MAX AMP is evaluated
        # ----------------------------------------------------------- Picker
        PickSample, PhaseInfo, _CF = pk_baer(tr.data, df, tdownmax_NEW,
                                             tupevent_NEW, thr1, thr2,
                                             preset_len_NEW, p_dur_NEW,
                                             return_cf=True)
        PickTime = PickSample/df    # convert pick from samples
                                    # to seconds (Absolute from first sample)
        PhaseInfo = str(PhaseInfo).strip()
        logger.debug("%s - %s" % (tr.stats.starttime+PickTime, PhaseInfo))
        # ------------------------------------------------------------- Save
        if PhaseInfo != '':  # Valid Pick
            self._storepick(it,           # first is keydict, second is it info
                            iteration=it,
                            pickUTC=tr.stats.starttime+PickTime,
                            bk_info=PhaseInfo)
        else:
            self._storepick(it,           # first is keydict, second is it info
                            iteration=it,
                            pickUTC=None,
                            bk_info=None)

    def AIC(self,
            useraw=False,
            aroundpick=None,
            wintrim_noise=1.0,
            wintrim_sign=1.0):
        """
        This method is defining an AIC picker
        to detect the right on-phase timing of a phase
        on a time-window defined by wintrim (seconds) before
        and after the pick. The AIC picker is much more precise than
        Baer to detect the right sample.

        IN:
            aroundpick: Must be an UTCDAteTime object. or set None/False

        OUT:
            pickTime_UTC, AIC, idx

        REFERENCES:
        - Kalkan, E. (2016). "An automatic P-phase arrival time picker",
          Bull. of Seismol. Soc. of Am., 106, No. 3,
          doi: 10.1785/0120150111
        - Akaike, H. (1974). A new look at the statistical model
          identification, Trans. Automat. Contr. 19, no. 6, 716–723,
          doi: 10.1109/TAC.1974.1100705

        """

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Inner Methods
        def AICcf(td, win=None):
            """
            This method will return the index of the minimum AIC
            carachteristic function.

            td must be a  `numpy.ndarray`
            """
            # --------------------  Creation of the carachteristic function
            # AIC(k)=k*log(variance(x[1,k]))+(n-k+1)*log(variance(x[k+1,n]))
            AIC = np.array([])
            for ii in range(1, len(td)):
                with np.errstate(divide='raise'):
                    try:
                        var1 = np.log(np.var(td[0:ii]))
                    except FloatingPointError:  # if var==0 --> log is -inf
                        var1 = 0.00
                    #
                    try:
                        var2 = np.log(np.var(td[ii:]))
                    except FloatingPointError:  # if var==0 --> log is -inf
                        var2 = 0.00
                #
                val1 = ii*var1
                val2 = (len(td)-ii-1)*var2
                AIC = np.append(AIC, (val1+val2))
            # -------------------- New idx search (avoid window's boarders)
            # (ascending order min->max) OK!
            idx = sorted(range(len(AIC)), key=lambda k: AIC[k])[0]

            # --- OLD (here for reference)
            # idxLst = sorted(range(len(AIC)), key=lambda k: AIC[k])
            # if idxLst[0]+1 not in (1, len(AIC)):  # need index. start from 1
            #     idx = idxLst[0]+1
            # else:
            #     idx = idxLst[1]+1

            # --- REALLY OLD  idx search (here for reference)
            # idx_old=int(np.where(AIC==np.min(AIC))[0])+1
            # ****   +1 order to make multiplications
            # **** didn't take into account to minimum at the border of
            # **** the searching window
            return idx, AIC

        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% MainAIC

        # --- Select trace 22022019 --> v2.1.6
        if useraw:
            self._setworktrace(self.wc, "RAW")  # sometimes is better for AIC
        else:
            self._setworktrace(self.wc, "PROC")
        #
        if not isinstance(self.wt, Trace):
            logger.error("Input trace is not a valid ObsPy trace: %s" %
                         type(self.wt))
            raise TypeError

        # Select TraceSlice:
        tr = self.wt.copy()
        if aroundpick:
            tr.trim(aroundpick - wintrim_noise,
                    aroundpick + wintrim_sign,
                    nearest_sample=True)
            td = tr.data
        else:
            # Entire stream
            td = tr.data

        # Get only the minimum of The CF
        idx, aicfun = AICcf(td)
        # -------------------- OUT
        # time= NUMsamples/df OR NUMsamples*dt
        logger.debug("AIC sample: %r" % idx)
        if aroundpick:
            pickTime_UTC = tr.stats.starttime + (idx * tr.stats.delta)
        else:
            pickTime_UTC = tr.stats.starttime + (idx * tr.stats.delta)

        # #MB nextline
        # from quake.plot import plot_QUAKE_CF   #MB
        # plot_QUAKE_CF(Stream(traces=tr),
        #                   {'AIC': aicfun},
        #                   chan="*Z",
        #                   picks={'aic_p': idx},
        #                   inax=None,
        #                   normalize=True,
        #                   show=True)

        return pickTime_UTC, aicfun, idx

    def evaluatePick_BK(self, pkey):
        """
        This method evaluates and analize the amplitude and
        Signal 2 Noise ratio of the time window around picks.
        CMN.TW defines it

        *** NB: this method evaluate only the actual ITERATION PICK !!!

        NB: with trim prob. error, no warning if start or end
            trace reached
            CMN.TW could be improved later in Config file (and passed by Main)

        NB: CMN.TW should be equal to the lowest class time uncertainties (old)
        NB: CMN.TW should be tuned with the type of regional/local/teleseismic ev.
        NB: Sequence of iteration log //
            EVENTID,STATION,ITERATION,PICK,PAR_1,TEST_1,TEST_2
        """
        testResults = []
        self._setworktrace(self.wc, "PROC")  # ALWAYS PROCESSED
        # ------------------------------------------- logging
        # LOG_ID.write(('%s'+CMN.FSout+'%s'+CMN.FSout+'%d'+CMN.FSout+'%s'+os.linesep )%(
        #              InTrace.stats['BaIt_DICT']['EVENTID'],
        #              InTrace.stats.station,
        #              it,
        #              InTrace.stats['BaIt_DICT'][str(it)]['baerpick'].strftime(CMN.GMTout_FMT) ))

        # ----------------------- Perform TESTS + Append to LOG info needed
        # *** NB always give a copy of input trace to the TEST!
        # *** NB "InTrace.copy(),it,CMN,LOG_ID" should be mandatory for every TEST!
        if self.pick_test:
            logger.info("Pick Evaluation: %s" % str(pkey))
            # sorting in alphabetical order the testfunctions
            sortedkeys = sorted(self.pick_test, key=str.lower)
            for _kk in sortedkeys:
                try:
                    testFunction = getattr(BCT, _kk)
                except AttributeError:
                    raise BE.MissingAttribute()
                #
                (verdict, testout) = testFunction(self.wt.copy(),
                                                  self.baitdict[str(pkey)],
                                                  *self.pick_test[_kk])
                testResults.append(verdict)
                self.baitdict[str(pkey)]["evaluatePick_tests"][_kk] = (verdict,
                                                                       testout)

        # ------------------------------- TestResult CHECK + LOG + exit
        if testResults:
            # *** NB: all test must be passed for pick to be accepted
            if False in testResults:
                logger.info(' '*4+'Pick:  * Rejected *')
                return False
            else:
                ######################################
                # HERE PUTS THE WEIGHTING CLASS SCHEME
                ######################################
                logger.info(' '*4+'Pick:    Accepted')
                return True
        else:
            # *** NB if user doesn't want CustomTests, it's because trust ObsPy BK
            # ***    It's his choice!!
            # *** It's like the default is set to True
            logger.info(" NO Pick Evaluation set --> accept anyway")
            return True

    def getTruePick(self, idx=0, picker="BK", format4quake=False):
        """
        Method to extract infor from self.baitdic
         - idx: the ordered index of validate pick (TRUE)
         - picker: define the picktime of TRUE pick to be analyzed and
                   used in the sorting process
         - format4quake: return UTCDateTIme pick (related to picker)
                         and pickinfo (always from BK algorithm)

        """
        if picker.lower() not in ("bk", "aic"):
            raise BE.BadKeyValue()
        #
        tmplst = []  # list of valid tuple [0] pickkey [1] UTCpick
        # populate
        for _kk in self.baitdict.keys():
            if self.baitdict[_kk]['evaluatePick']:
                if picker.lower() == 'bk':
                    tmplst.append((_kk, self.baitdict[_kk]['pickUTC']))
                elif picker.lower() == 'aic':
                    tmplst.append((_kk, self.baitdict[_kk]['pickUTC_AIC']))
        # extract
        tmplst.sort(key=itemgetter(1))
        try:
            pd = self.baitdict[tmplst[idx][0]]
        except IndexError:
            if format4quake:
                return None, None
            else:
                return None
        #
        if format4quake:
            if picker.lower() == 'bk':
                return pd['pickUTC'], pd['bk_info']
            elif picker.lower() == 'aic':
                return pd['pickUTC_AIC'], pd['bk_info']
        else:
            return pd

    def plotPicks(self, plotraw=False, **kwargs):
        """
        Wrapper method that calls theplotting routine
        Returns:
         - fig handle
         - ax tuples (more than one possible)
        """
        # Create CF always on PROC trace
        self._setworktrace(self.wc, "PROC")
        cf = self.wt.copy()
        cf.data = BCT._createCF(cf.data)

        # select time series
        if plotraw:
            self._setworktrace(self.wc, "RAW")
        else:
            self._setworktrace(self.wc, "PROC")
        #
        fig, ax = BP.plotBait(self.wt, cf, self.baitdict, **kwargs)
        return fig, ax

    # def evaluatePick_BK_POST(self):
    #     """
    #     This method will call and perform several User-Custom test
    #     for the acceptance or denial of a certain pick after a full-circle
    #     picking has been done to the waveform. Every function called in this
    #     function will take care to evaluate negative the picks

    #     RETURN: a list of valid picks tuple with [0] UTCDateTime, [1] iteration

    #     """
    #     if self.post_test:
    #         logger.info("Post-Picking Evaluation")

    #     # Will be a list of tuple with [0] UTCDateTime, [1] iteration
    #     validPicks = []
    #     for _xx in self.baitdict.keys():
    #         if self.baitdict[_xx]['evaluatePick']:
    #             validPicks.append(self.baitdict[_xx]['pickUTC'], _xx)

    #     if validPicks:
    #         # sorting in alphabetical order the testfunctions
    #         sortedkeys = sorted(self.baitdict, key=str.lower)
    #         for _kk in sortedkeys:
    #             try:
    #                 testFunction = getattr(BCT, _kk)
    #             except AttributeError:
    #                 raise BE.MissingAttribute()
    #             #
    #             testout = testFunction(self.wt, validPicks,
    #                                    *self.post_dict[_kk])
    #             self.baitdictPOST["evaluatePick_tests"][_kk] = testout
    #     else:
    #         logger.info("NO Post-Picking Evaluation set")
