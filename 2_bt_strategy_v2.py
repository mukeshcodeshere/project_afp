import backtrader as bt 


def buy_and_hold_strategy():
    cerebro = bt.Cerebro()
    cerebro.add_signal(bt.SIGNAL_LONG, bt.indicators.AllN)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class CloseVsSma(bt.Indicator):
    lines = ('signal',)

    params = (
        ('window', 30),
        ('gap', .1),
        ('mdir', 'down')
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.p.window)

    def next(self):
        sma_share = self.data/self.sma
        if sma_share < 1-self.p.gap:
            self.lines.signal[0] = 1
        elif sma_share > 1+self.p.gap:
            self.lines.signal[0] = -1
        else:
            self.lines.signal[0] = 0


def close_vs_sma_strategy():
    cerebro = bt.Cerebro()

    cerebro.add_signal(bt.SIGNAL_LONG, CloseVsSma, window=15, gap=.05)
    cerebro.add_signal(bt.SIGNAL_LONGEXIT, CloseVsSma, window=60, gap=.05)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class MeanReversion(bt.Indicator):
    lines = ('signal',)

    params = (
        ('window_small', 30),
        ('window_large', 90),
    )

    def __init__(self):
        self.sma = bt.indicators.SMA(period=self.p.window_small)
        self.lma = bt.indicators.SMA(period=self.p.window_large)

    def next(self):
        if self.sma < self.lma:
            self.lines.signal[0] = 1
        elif self.sma > self.lma:
            self.lines.signal[0] = -1


def mean_reversion_strategy():
    cerebro = bt.Cerebro()

    cerebro.add_signal(bt.SIGNAL_LONG, MeanReversion)
    cerebro.add_signal(bt.SIGNAL_LONGEXIT, MeanReversion)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class PrevPeak(bt.Indicator):
    lines = ('signal',)

    params = (
        ('window', 90),
        ('max_thresold', .9),
        ('min_thresold', 1.1),
    )

    def __init__(self):
        self.win_max = bt.indicators.Highest(period=self.p.window)
        self.win_min = bt.indicators.Lowest(period=self.p.window)

    def next(self):
        max_percents = self.data/self.win_max
        min_percents = self.data/self.win_min

        if max_percents > self.p.max_thresold:
            self.lines.signal[0] = -1
        elif min_percents < self.p.min_thresold:
            self.lines.signal[0] = 1


def prev_peak_strategy():
    cerebro = bt.Cerebro()

    cerebro.add_signal(bt.SIGNAL_LONG, PrevPeak)
    cerebro.add_signal(bt.SIGNAL_LONGEXIT, PrevPeak)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class Random(bt.Indicator):
    lines = ('signal',)

    params = (
        ('prob', .1),
    )

    def next(self):
        if random.random() < self.p.prob:
            self.lines.signal[0] = 1
        elif random.random() < self.p.prob:
            self.lines.signal[0] = -1


def random_strategy():
    cerebro = bt.Cerebro()

    #cerebro.addstrategy(RandomStrategy)
    cerebro.add_signal(bt.SIGNAL_LONG, Random)
    cerebro.add_signal(bt.SIGNAL_LONGEXIT, Random)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class PrevPeakNoDrop(bt.Strategy):
    def __init__(self):
        self.acquired = False
        self.prev_peak = PrevPeak()

    def next(self):
        if not self.acquired:
            if self.prev_peak > 0:
                self.buy()
                self.buy_price = self.data[0]
                self.acquired = True
        elif self.acquired:
            if self.prev_peak < 0 and self.buy_price < self.data:
                self.close()
                self.acquired = False


def prev_peak_nodrop_strategy():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(PrevPeakNoDrop)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class CloseVsSmaNoDrop(bt.Strategy):
    def __init__(self):
        self.acquired = False
        self.csma_down = CloseVsSma(window=15, gap=.05)
        self.csma_up = CloseVsSma(window=15, gap=.05)

    def next(self):
        if not self.acquired:
            if self.csma_down > 0:
                self.buy()
                self.buy_price = self.data[0]
                self.acquired = True
        elif self.acquired:
            if self.csma_up < 0 and self.buy_price < self.data:
                self.close()
                self.acquired = False


def close_vs_sma_nodrop_strategy():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(CloseVsSmaNoDrop)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class MeanReversionNoDrop(bt.Strategy):
    def __init__(self):
        self.acquired = False
        self.mr = MeanReversion()

    def next(self):
        if not self.acquired:
            if self.mr > 0:
                self.buy()
                self.buy_price = self.data[0]
                self.acquired = True
        elif self.acquired:
            if self.mr < 0 and self.buy_price < self.data:
                self.close()
                self.acquired = False


def mean_reversion_nodrop_strategy():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(MeanReversionNoDrop)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro


class AntiDrop(bt.Strategy):
    def __init__(self):
        self.initial_buy = False
        self.acquired = False
        self.mr_down = MeanReversion(window_small=5, window_large=30)
        self.mr_up = MeanReversion(window_small=30, window_large=90)

    def next(self):
        if not self.initial_buy:
            self.buy()
            self.acquired = True
            self.initial_buy = True
        elif not self.acquired:
            if self.data[0] > self.sell_price:
                self.buy()
                self.acquired = True
            if self.mr_up > 0:
                self.buy()
                self.acquired = True
        elif self.acquired:
            if self.mr_down < 0:
                self.close()
                self.acquired = False
                self.sell_price = self.data[0]


def anti_drop_strategy():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(AntiDrop)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=100)
    return cerebro