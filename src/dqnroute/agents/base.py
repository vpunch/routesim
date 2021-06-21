from typing import Callable

import networkx as nx

from ..messages import *


class MsgHandler:
    # Базовый агент, который обрабатывает пришедшие сообщения

    def __init__(self,
                 id: int,
                 get_time: Callable[[], int],
                 router_graph: nx.DiGraph):
        super().__init__()  # конструктор для reward agent

        self.id = id
        self.get_time = get_time
        self.router_graph = router_graph

    @property
    def out_nbrs(self):
        return [nbr for _, nbr in self.router_graph.out_edges(self.id)]

    @property
    def in_nbrs(self):
        return [nbr for nbr, _ in self.router_graph.in_edges(self.id)]

    @property
    def all_nbrs(self):
        return self.in_nbrs + self.out_nbrs

    def handle(self, msg: Message) -> list[Message]:
        if isinstance(msg, InMsg):
            sender = msg.from_node
            msg = msg.inner_msg

            if isinstance(msg, PkgMsg):
                pkg = msg.pkg

                if pkg.dst == self.id:
                    # Пакет пришел нам

                    return [PkgReceivedMsg(pkg)]
                else:
                    # Пакет не наш, передаем его дальше

                    next_node, resp = self.route(sender, pkg)
                    return [OutMsg(self.id, next_node, PkgMsg(pkg))] + resp
            elif isinstance(msg, ServiceMsg):
                return self.handle_service_msg(sender, msg)
        elif isinstance(msg, InitMsg):
            return self.init(msg.config)
        elif isinstance(msg, AddLinkMsg):
            return self.add_link(**msg.content)
        elif isinstance(msg, RemoveLinkMsg):
            return self.remove_link(**msg.content)
        else:
            raise UnsupportedMsgType(msg)

    def init(self, config) -> list[Message]:
        return []

    def add_link(self,
                 node: int,
                 direction: str,
                 edge_data: dict) -> list[Message]:
        #if direction != 'out':  # node ->(in) self
        #    self.router_graph.add_edge(node, self.id, **edge_data)

        if direction != 'in':  # node <-(out) self
            self.router_graph.add_edge(self.id, node, **edge_data)

        return []

    def remove_link(self, node: int, direction: str) -> list[Message]:
        #if direction != 'out':
        #    self.router_graph.remove_edge(node, self.id)

        if direction != 'in':
            self.router_graph.remove_edge(self.id, node)

        return []

    def route(self, sender: int, pkg: Package) -> tuple[int, list[Message]]:
        raise NotImplementedError()

    def handle_service_msg(self,
                           sender: int,
                           msg: ServiceMsg) -> list[Message]:
        raise UnsupportedMsgType(msg)


class RewardAgent:
    def __init__(self):
        # Пакеты, для которых ожидается получение награды
        self.pending_pkgs = {}

    # Регистрация пакета
    def register_recent_pkg(self,
                            pkg: Package,
                            estim: float,
                            data) -> RewardMsg:
        reward_data = self.__get_reward_data()

        # Ожидаем награду от следующего узла
        self.pending_pkgs[pkg.id] = (reward_data, data)

        # Выдаем награду предыдущему узлу
        return self.__create_reward(pkg.id, estim, reward_data)

    # Расчет награды
    def receive_reward(self, msg: RewardMsg):
        rdata_old, data = self.pending_pkgs.pop(msg.pkg_id)
        reward = self.__compute_reward(msg, rdata_old)
        return reward, data

    def __compute_reward(self, msg: RewardMsg, time_sent):
        time_received = msg.reward_data
        return msg.estim + (time_received - time_sent)

    def __create_reward(self,
                        pkg_id: int,
                        estim: float,
                        time_sent) -> RewardMsg:
        return RewardMsg(pkg_id, estim, time_sent)

    def __get_reward_data(self):
        return self.get_time()
