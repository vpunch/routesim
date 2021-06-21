from copy import deepcopy

import networkx as nx

from .base import *
from ..messages import *


class LinkStateRouter(MsgHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.seq_num = 0
        self.announs = {}

    def init(self,
             config: dict) -> list[Message]:
        msgs = super().init(config)
        return msgs + self.__announce_state()

    def handle_service_msg(self,
                           sender: int,
                           msg: ServiceMsg) -> list[Message]:
        if isinstance(msg, StateAnnounMsg):
            if self.__proc_announ(msg):
                # Если анонса еще не было

                # Кажется, что тут в кольце отправитель нового состояния
                # получает его и обрабатывает

                # Разослать пришедшее состояние соседям
                nbrs = self.out_nbrs
                nbrs.remove(sender)
                return [OutMsg(self.id, nbr, msg) for nbr in nbrs]

            return []
        else:
            return super().handle_service_msg(sender, msg)

    def add_link(self,
                 node: int,
                 direction: str,
                 edge_data: dict) -> list[Message]:
        resp = super().add_link(node, direction, edge_data)
        # Разослать новое состояние соседям
        return resp + self.__announce_state()

    def remove_link(self,
                    node: int,
                    direction: str) -> list[Message]:
        resp = super().remove_link(node, direction)
        return resp + self.__announce_state()

    def route(self, sender: int, pkg: Package) -> tuple[int, list[Message]]:
        # Найти кратчайший путь и взять оттуда соседа
        path = nx.dijkstra_path(self.router_graph, self.id, pkg.dst)
        return path[1], []

    def __announce_state(self) -> list[Message]:
        self.seq_num += 1
        state = self.router_graph.adj[self.id]
        announ = StateAnnounMsg(self.id, self.seq_num, state)
        resp = [OutMsg(from_node=self.id, to_node=v, inner_msg=announ)
                for v in self.out_nbrs]

        return resp

    def __proc_announ(self, msg: StateAnnounMsg) -> bool:
        if msg.node not in self.announs or \
                self.announs[msg.node].seq < msg.seq:
            self.announs[msg.node] = msg
            self.__proc_new_announ(msg.node, msg.state)

            return True

        return False

    def __proc_new_announ(self, node: int, state) -> bool:
        # Удалить всех соседей узла

        # Добавить соседей из состояния
        edges = list(self.router_graph.edges(node))
        self.router_graph.remove_edges_from(edges)

        for nbr, edge_data in state.items():
            self.router_graph.add_edge(node, nbr, **edge_data)
