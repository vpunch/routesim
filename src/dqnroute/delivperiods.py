import pandas as pd


class DelivPeriods:
    def __init__(self, period_dur, aggregators):
        columns = ['time'] + list(aggregators.keys())
        self.periods = pd.DataFrame(columns=columns)

        # Длительность одного интервала
        self.period_dur = period_dur
        # Функции для формирования нового значения на основе предыдущего
        self.aggregators = aggregators

    def __register(self, period_num, deliv_time):
        aggr_cols = self.periods.columns[1:]

        try:
            aggr_data = self.periods.loc[period_num, aggr_cols]
        except KeyError:  # новый интервал
            aggr_data = [None] * len(aggr_cols)

        aggr_data_new = [
            self.aggregators[col](value, deliv_time)
            for col, value in zip(aggr_cols, aggr_data)
        ]

        time = period_num * self.period_dur
        self.periods.loc[period_num] = [time] + aggr_data_new

    def register(self, start_time: int, cur_time: int):
        deliv_time = cur_time - start_time
        period_num = start_time // self.period_dur

        self.__register(period_num, deliv_time)

    def get_periods(self):
        return self.periods.sort_index()


# Обработка начального значения
def wrap_aggr(fun, init=None):
    def wrapper(value, delta):
        if value is None:
            return init if init is not None else delta

        return fun(value, delta)

    return wrapper


def get_aggregator(name):
    if name == 'sum':
        return wrap_aggr(lambda value, delta: value + delta)
    elif name == 'count':
        return wrap_aggr(lambda value, _: value + 1, 1)
    elif name == 'max':
        return wrap_aggr(max)
    elif name == 'min':
        return wrap_aggr(min)
    else:
        raise Exception('Unknown aggregator function ' + name)


def create_periods(period_dur, aggr_names):
    aggregators = {name: get_aggregator(name) for name in aggr_names}

    return DelivPeriods(period_dur, aggregators)
