from .base import *
from ..messages import *


class QRouter(MsgHandler, RewardAgent):
    def __init__(self,
                 learning_rate: float,
                 **kwargs):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate
        self.policy = 'strict'
        self.init_estim = 10

        # Q-таблица:
        #     y1 y2 y3  <- выходные соседи (ys)
        # d1   0 10 10
        # d2  10  0 10
        # d3  10 10  0
        # ^
        # `- конечные узлы (ds)
        self.Q = {}
        for d in self.all_nbrs:
            self.check_dst(d)

    def add_link(self,
                 node: int,
                 direction: str,
                 edge_data: dict) -> list[Message]:
        # Добавить соседа в список соседей
        resps = super().add_link(node, direction, edge_data)

        # Добавить соседа в Q-таблицу
        if direction == 'out':
            self.check_nbr(node)

        # Если выходной сосед был удален, то оставить его в таблице, но не
        # учитывать при маршрутизации

        return resps

    def route(self, sender: int, pkg: Package) -> tuple[int, list[Message]]:
        self.check_dst(pkg.dst)

        Qs = self.actual_Q(pkg.dst)
        # Получить соседа с минимальной оценкой
        next_node, estim = min(Qs.items(), key=lambda x: x[1])

        reward_msg = self.register_recent_pkg(pkg, estim, pkg.dst)

        resp = []
        if sender != -1:  # первый узел в пути пакета
            resp.append(OutMsg(from_node=self.id,
                               to_node=sender,
                               inner_msg=reward_msg))

        return next_node, resp

    def handle_service_msg(self, y: int, msg: ServiceMsg) -> list[Message]:
        if isinstance(msg, RewardMsg):
            # Получили обратную связь

            estim_new, d = self.receive_reward(msg)
            estim_delta = estim_new - self.Q[d][y]
            self.Q[d][y] += estim_delta * self.learning_rate

            return []
        else:
            return super().handle_service_msg(y, msg)

    def check_dst(self, d):
        if d not in self.Q:
            self.Q[d] = {y: 0 if d == y else self.init_estim
                         for y in self.out_nbrs}

    # Только выходные соседи
    def check_nbr(self, nbr):
        for d, ys in self.Q.items():
            if nbr not in ys:
                ys[to] = 0 if d == nbr else self.init_estim

    def actual_Q(self, d: int) -> dict[int, float]:
        # Получить из Q-таблицы оценки только для нужного узла через актуальных
        # соседей

        Qa = {}
        for n in self.out_nbrs:
            Qa[n] = self.Q[d][n]

        return Qa
