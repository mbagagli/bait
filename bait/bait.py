"""
This module is extracted from BaIt v2.1.0.
This iterative picking algorithm is hopefully performing better
"""

# lib for MAIN
import logging
# lib for BAIT
from bait import bait_errors as BE
from bait import bait_customtests as BCT
from obspy.signal.trigger import pk_baer
from operator import itemgetter
# lib for Errors
from obspy.core.trace import Trace

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

    """
    def __init__(self,
                 stream,
                 channel="*Z",
                 max_iter=5,
                 opbk_main={},
                 opbk_aux={},
                 test_pickvalidation={},
                 test_postvalidation={}):
        self.st = stream
        self.wc = channel
        self.maxit = max_iter
        self.opbk_main = opbk_main
        self.opbk_aux = opbk_aux
        self.wt = self._setworktrace(self.wc)   # workingtrace
        #
        self.pick_test = test_pickvalidation
        self.post_test = test_postvalidation
        self.baitdict = {}
        self.baitdict_keys = ('pickUTC',
                              'bk_info',
                              'iteration',
                              'evaluatePick',
                              'evaluatePick_tests')

    def _sec2sample(value, df):
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
        self.baitdict[str(addkey)] = {_kk: None for _kk in self.baitdict_keys}
        for _kk in self.pick_test.keys():
            self.baitdict[str(addkey)]["evaluatePick_tests"][_kk] = None
        for _kk in self.post_test.keys():
            self.baitdict[str(addkey)]["evaluatePickPost_tests"][_kk] = None

    def _storepick(self, storekey, pickUTC=None, pickInfo=None):
        """
        This is the method that is called FIRST by bait algorithm.
        Is called by `picker` method.

        *** This method take care to initialize missing iteraion
            dict keys in self.baitdict

        """
        if str(storekey) not in self.baitdict.keys():
            self._createbaitdictkey(str(storekey))
        #
        self.baitdict[str(storekey)]['iteration'] = storekey
        if pickUTC and pickInfo:
            self.baitdict[str(storekey)]['pickUTC'] = pickUTC
            self.baitdict[str(storekey)]['bk_info'] = pickInfo
        else:
            self.baitdict[str(storekey)]['pickUTC'] = None
            self.baitdict[str(storekey)]['bk_info'] = None

    def _setworktrace(self, channel, **kwargs):
        """
        Private method to change pointer of working trace for picker.
        Use obspy.Stream.select function
        """
        self.wt = self.st.select(channel=channel, **kwargs)
        self.wc = channel

    def _getbaitdict(self):
        return self.baitdict

    def _getpick4quake(self, pkey="valid"):
        """
        This method returns the validated correct pick in order to be
        appended into QUAKE.pickContainer class.

        KEYS for QUAKE:
                    'polarity',       # (str/None)
                    'onset',          # (str/None)
                    'weight',         # (float/None)
                    'pickclass',      # (int/None)
                    'timeUTC_pick',   # (UTCDateTime/None)
                    'timeUTC_early',  # (UTCDateTime/None)
                    'timeUTC_late',   # (UTCDateTime/None)

        """
        try:
            out = {'polarity': self.baitdict[pkey]["polarity"],
                   'onset': self.baitdict[pkey]["onset"],
                   'weight': None,
                   'pickclass': self.baitdict[pkey]["class"],
                   'timeUTC_pick': self.baitdict[pkey]["pick"],
                   'timeUTC_early': None,
                   'timeUTC_late': None
                   }
        except KeyError:
            # No pick found with the given key
            out = {'polarity': None,
                   'onset': None,
                   'weight': None,
                   'pickclass': None,
                   'timeUTC_pick': None,
                   'timeUTC_early': None,
                   'timeUTC_late': None
                   }
        return out

    def picker(self,
               workTrace,
               it,
               tdownmax,
               tupevent,
               thr1,
               thr2,
               preset_len,
               p_dur):
        """
        This method is a wrapper to the autopicker Baer&Kradofler
        RETURNS: type::obspy.Trace
            where:
                - workTrace.stats['BaIt_DICT'][str(it)]['baerpick'] :: UTCDateTime
                - workTrace.stats['BaIt_DICT'][str(it)]['info'] :: str
        The dict keys are 'None' type if no pick found
        """
        if not isinstance(workTrace, Trace):
            raise BE.BadInstance()
        # -------------------------------------------------------- Cut Trace
        tr = workTrace.copy()
        if it > 1:
            tr = tr.trim(workTrace.stats['BaIt_DICT'][str(it-1)]['baerpick'],
                         workTrace.stats.endtime)
        # ------------------------------------------------- v.1.1 sample2sec
        # mainly we change the input from BaIt_Config in SECONDS and convert here in SAMPLES
        # Python3 round(float)==int // Python2 round(float)==float --> int(round(... to have compatibility
        df = workTrace.stats.sampling_rate
        preset_len_NEW = self._sec2sample(preset_len, df)
        tupevent_NEW = self._sec2sample(tupevent, df)            # tupevent: should be the inverse of high-pass
                                                                 #           freq or low freq in bandpass
        tdownmax_NEW = self._sec2sample(tdownmax, df)            # Half of tupevent
        p_dur_NEW = self._sec2sample(p_dur, df)                  # time-interval in which MAX AMP is evaluated
        # ----------------------------------------------------------- Picker
        PickSample, PhaseInfo = pk_baer(tr.data, df, tdownmax_NEW,
                                        tupevent_NEW, thr1, thr2,
                                        preset_len_NEW, p_dur_NEW)
        PickTime = PickSample/df    # convert pick from samples
                                    # to seconds (Absolute from first sample)
        PhaseInfo = str(PhaseInfo).strip()
        # ------------------------------------------------------------- Save
        if PhaseInfo != '':  # Valid Pick
            self._storepick(it, it,  # first is keydict, second is it info
                            pickUTC=tr.stats.starttime+PickTime,
                            pickInfo=PhaseInfo)
        else:
            self._storepick(it, it,  # first is keydict, second is it info
                            pickUTC=None,
                            pickInfo=None)

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
                self.picker(self.wt, ITERATION, **self.opbk_main)
            else:
                self.picker(self.wt, ITERATION, **self.opbk_aux)
            #
            if self.baitdict[str(ITERATION)]['pickUTC']:
                # pick found -> evaluate
                if self.pick_test:
                    # Check if PICK_BK is valid
                    self.baitdict[str(ITERATION)]['evaluate'] = (
                        self.evaluatePick_BK(self.wt, ITERATION, GB, MAIN, ERR)
                    )
                if self.baitdict[str(ITERATION)]['evaluate']:
                    # If pick is valid and user wants AIC --> call AIC picker

                    # ----- If you want to add an AIC picker over the iterationBK
                    # if GB.AIC_PICKER: # v1.1.5
                    #     CopiedTrace.stats['BaIt_DICT'][str(ITERATION)]['aicpick'],AIC_CF,_=BP.bAIC(
                    #         CopiedTrace,CopiedTrace.stats['BaIt_DICT'][str(ITERATION)]['baerpick'],
                    #         wavedata=GB.AIC_DATA,wintrim_noise=GB.AICwin_noise,wintrim_signal=GB.AICwin_signal)
                    VALIDPICKS = True

            else:     # no pick ->skip to next trace
                logger.error(('~~~ No pick @ Iteration %d') % (ITERATION))
                break
        # Post - Picking
        if not VALIDPICKS:
            logger.error("*** No valid pick found")
        else:
            if self.post_test:
                self.evaluatePick_BK_POST()
        #
        return True

    def evaluatePick_BK(self, pkey, CMN, LOG_ID, ERR_ID):
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
                self.baitdict[str(pkey)]["evaluatePick_tests"][_kk] = testout

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

    # def getTruePick(self, idx=0):
    #     """ return the baitdict idx key among the sorted TRUE pick """
    #     tmplst = []  # list of valid tuple [0] pickkey [1] UTCpick
    #     # populate
    #     for _kk in self.baitdict.keys():
    #         if self.baitdict[_kk]['evaluatePick']:
    #             tmplst.append((_kk, self.baitdict[_kk]['pickUTC']))
    #     # extract
    #     tmplst.sort(key=itemgetter(1))
    #     try:
    #         return self.baitdict[tmplst[idx][0]]
    #     except IndexError:
    #         return None
