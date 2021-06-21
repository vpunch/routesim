import logging

import networkx as nx
from simpy import Environment, Event, Resource, Process

from .network import *
from ..messages import *
from ..delivperiods import *
from ..agents import *
from ..constants import *
from ..utils import *

logger = logging.getLogger(MAIN_LOGGER)


class RouterEnv(NodeEnv):
    def __init__(self,
                 env: Environment,
                 router,
                 node: int,
                 deliv_periods: DelivPeriods,
                 local_graph: nx.DiGraph,
                 pkg_process_delay: int):
        super().__init__(env, router)

        self.id = node
        self.deliv_periods = deliv_periods
        self.local_graph = local_graph
        self.pkg_process_delay = pkg_process_delay

        self.msg_proc_queue = Resource(self.env, capacity=1)

    def msg_event(self, msg: Message) -> Event:
        if isinstance(msg, (InitMsg, AddLinkMsg, RemoveLinkMsg)):
            # Достаточно только обработать роутером

            return self.env.event().succeed()
        elif isinstance(msg, OutMsg):
            # Отправляем сообщение

            return self.env.process(self.__edge_transfer(msg))
        elif isinstance(msg, InMsg):
            # Принимаем сообщение

            return self.env.process(self.__input_queue(msg))
        elif isinstance(msg, PkgReceivedMsg):
            # Пакет доставлен

            logger.debug((f'Package #{msg.pkg.id} received '
                          f'at node {self.id} at time {self.env.now}'))
            self.deliv_periods.register(msg.pkg.start_time, self.env.now)
            return self.env.event().succeed()
        else:
            return super().msg_event(msg)

    def __edge_transfer(self, msg):
        #print(self.local_graph.edges)
        #print(msg)
        edge_data = self.local_graph.edges[self.id, msg.to_node]
        nbr_router_env = self.local_graph.nodes[msg.to_node]['router_env']
        new_msg = InMsg(**msg.content)
        inner_msg = msg.inner_msg

        # Сервисные сообщения не засоряют канал
        if isinstance(inner_msg, ServiceMsg):
            nbr_router_env.receive(new_msg)
        elif isinstance(inner_msg, PkgMsg):
            pkg = inner_msg.pkg
            logger.debug(
                    f'Package #{pkg.id} hop: {msg.from_node} -> {msg.to_node}')

            bw = edge_data['bandwidth']
            with edge_data['resource'].request() as req:
                yield req
                yield self.env.timeout(pkg.size / bw)

            nbr_router_env.receive(new_msg)
        else:
            raise UnsupportedMessageType(inner_msg)

    def __input_queue(self, msg: InMsg):
        inner_msg = msg.inner_msg

        if isinstance(inner_msg, ServiceMsg):
            pass
        elif isinstance(inner_msg, PkgMsg):
            with self.msg_proc_queue.request() as req:
                yield req  # ждем обработки предыдущего сообщения
                yield self.env.timeout(self.pkg_process_delay)
        else:
            raise UnsupportedMessageType(inner_msg)


# Окружение для симуляции компьютерной сети
class ComputerNetEnv(NetworkEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        RouterClass = get_router_class(self.router_type)
        # (node, {nbr: edge_data, ...}) ...
        # Получаем исходящих соседей каждого узла
        for node, nbrs in self.graph.adjacency():
            # Специфичная конфигурация
            router_cfg = self.get_router_cfg(node)

            local_graph = self.graph.subgraph([node] + list(nbrs.keys()))

            router_graph = nx.DiGraph()
            for u, v in list(local_graph.edges):
                router_graph.add_edge(u, v, **self.__copy_edge_data(u, v))

            router = RouterClass(get_time=lambda: self.env.now,
                                 id=node,
                                 router_graph=router_graph,
                                 **router_cfg)

            # Окружение сети имеет граф сети, окружение роутера имеет
            # локальный подграф с исходящими соседями, роутер имеет копию
            # локального подграфа с исходящими соседями. Исходящие соседи в
            # компьютерной сети вактически являются всеми соседями. Не может
            # быть входящего линка без исходящего

            # Удаление линка не изменяет структуру графа, но изменяет список
            # соседей роутера

            self.graph.nodes[node]['router_env'] = RouterEnv(
                    self.env,
                    router,
                    node,
                    self.deliv_periods,
                    local_graph,
                    **self.run_params['settings']['router_env'])

        for _, router in self.graph.nodes.data('router_env'):
            router.receive(InitMsg({}))

    def __copy_edge_data(self, u, v):
        data = self.graph.edges[u, v].copy()
        del data['resource']
        return data

    def create_graph(self, run_params) -> nx.DiGraph:
        # Входящие и исходящие ребра создаются и обрываются парами в
        # компьютерной сети

        def parse_edge(edge):
            params = edge.copy()
            params['weight'] = 1 / params['bandwidth']
            u = params.pop('u')
            v = params.pop('v')
            return u, v, params

        graph = nx.DiGraph()
        for edge in run_params['network']:
            u, v, params = parse_edge(edge)
            graph.add_edge(
                    u, v, resource=Resource(self.env, capacity=1), **params)
            graph.add_edge(
                    v, u, resource=Resource(self.env, capacity=1), **params)

        return graph

    def run_process(self, random_seed=None):
        if random_seed is not None:
            set_random_seed(random_seed)

        pkg_distr = self.run_params['settings']['pkg_distr']
        pkg_id = 1
        for seq in pkg_distr['sequence']:
            if 'action' in seq:
                action = seq['action']
                pause = seq['pause']
                u = seq['u']
                v = seq['v']

                # Можно только обрывать и восстанавливать соединение. Добавлять
                # новые нельзя.

                if action == 'break_link':
                    self.graph.nodes[u]['router_env'].receive(RemoveLinkMsg(v))
                    self.graph.nodes[v]['router_env'].receive(RemoveLinkMsg(u))
                elif action == 'restore_link':
                    self.graph.nodes[u]['router_env'].receive(AddLinkMsg(
                            v, edge_data=self.__copy_edge_data(u, v)))
                    self.graph.nodes[v]['router_env'].receive(AddLinkMsg(
                            u, edge_data=self.__copy_edge_data(v, u)))

                yield self.env.timeout(pause)
            else:
                delta = seq['delta']
                all_nodes = list(self.graph.nodes)
                sources = seq.get('sources', all_nodes)
                dests = seq.get('dests', all_nodes)

                for i in range(0, seq['pkg_number']):
                    src = random.choice(sources)
                    dst = random.choice(dests)
                    pkg = Package(pkg_id, 1024, dst, self.env.now, None)

                    logger.debug(
                            (f'Sending random package #{pkg_id} from {src}'
                             f'to {dst} at time {self.env.now}'))
                    self.graph.nodes[src]['router_env'].receive(
                            InMsg(-1, src, PkgMsg(pkg)))

                    pkg_id += 1
                    yield self.env.timeout(delta)
