import networkx as nx
from simpy import Environment, Event, Interrupt

from ..delivperiods import *
from ..messages import *
from ..agents import *


# Окружение узла
class NodeEnv:
    def __init__(self, env: Environment, handler: MsgHandler):
        self.env = env
        self.handler = handler

        self.delayed_msgs = {}

    # Принять сообщение
    def receive(self, msg: Message) -> Event:
        #print(msg, self.id)
        self.env.process(self.__process_msg(msg))

    def __process_msg(self, msg: Message):
        # Входящие сообщения могут стоять в очереди на обработку,
        # поэтому нужно ждать
        # Обработка входящих сообщений занимает время, одновременно нельзя
        yield self.msg_event(msg)

        # Обработать сообщение и передать ответы дальше
        # Ответы передаются параллельно, ждать соответствующие события
        # не нужно
        for msg in self.handler.handle(msg):
            #print(msg)
            self.msg_event(msg)

    # Получить событие для сообщения
    def msg_event(self, msg: Message) -> Event:
        if isinstance(msg, DelayedMsg):
            # Откладываем сообщение

            return self.env.process(self.__delayed_event(msg))
        elif isinstance(msg, InterruptDelayMsg):
            # Возобновляем обработку сообщения

            self.delayed_msgs[msg.delay_id].interrupt()
            return self.env.event().succeed()
        else:
            raise UnsupportedMsgType(msg)

    def __delayed_event(self, msg):
        delay_event = self.env.timeout(msg.delay)
        self.delayed_msgs[msg.id] = delay_event

        try:
            yield delay_event  # ждем задержку
        except Interrupt:
            pass

        del self.delayed_msgs[msg.id]

        yield self.msg_event(msg.inner_msg)  # ждем сообщение


# Окружение для семуляции сети
class NetworkEnv:
    def __init__(self,
                 run_params,
                 router_type: str,
                 deliv_periods: DelivPeriods):
        self.run_params = run_params
        self.router_type = router_type
        self.deliv_periods = deliv_periods

        self.env = Environment()
        self.graph = self.create_graph(run_params)

    def get_router_cfg(self, node):
        router_cfg = self.run_params['settings']['router'].get(
                self.router_type, {})

        if self.router_type == 'dqn':
            router_cfg['net_size'] = self.graph.number_of_nodes()

        if self.router_type == 'dqn-le':
            router_cfg['net_size'] = self.graph.number_of_nodes()

        return router_cfg

    def run(self, random_seed=None):
        self.env.process(self.run_process(random_seed))
        self.env.run()

    def create_graph(self, run_params) -> nx.DiGraph:
        raise NotImplementedError()

    def run_process(self, random_seed):
        raise NotImplementedError()
