import logging

import torch as tch
import numpy as np
import networkx as nx

from .base import *
from .linkstate import *
from ..constants import MAIN_LOGGER
from ..messages import *
from ..memory import *
from ..utils import *
from ..networks import *

MIN_TEMP = 0.5

logger = logging.getLogger(MAIN_LOGGER)


class DQNRouter(LinkStateRouter, RewardAgent):
    """
    A router which implements DQN-routing algorithm
    """
    def __init__(self,
                 net_size,
                 batch_size: int,
                 mem_capacity: int,
                 activation,
                 layers,
                 embeddings,
                 optimizer,
                 addit_inputs=[],
                 **kwargs):
        super().__init__(**kwargs)

        self.net_size = net_size
        self.batch_size = batch_size
        self.memory = Memory(mem_capacity)
        self.addit_inputs = addit_inputs

        self.brain = QNetwork(self.net_size,
                              layers,
                              activation,
                              embeddings,
                              addit_inputs)
        self.brain.restore()
        logger.info('Restored model ' + self.brain.label)
        #print(self.brain.label)

        # Нужно обновлять энкодер, когда меняется топология
        self.node_enc = node_encoder_class(
                embeddings['name'])(embeddings['dim'])

        self.optimizer = optim_class(
                optimizer['name'])(self.brain.parameters(), lr=optimizer['lr'])
        self.loss_func = tch.nn.MSELoss()

    def route(self, sender: int, pkg: Package) -> tuple[int, list[Message]]:
        states = self.__get_nbr_states(pkg.dst)
        pred = self.__predict(states).flatten()
        to_idx = soft_argmax(pred, MIN_TEMP)

        to = sorted(self.out_nbrs)[to_idx]

        estim = -np.max(pred)
        saved_state = [elem[to_idx] for elem in states if len(elem) != 0]
        reward = self.register_recent_pkg(pkg, estim, saved_state)

        return to, [OutMsg(self.id, sender, reward)] if sender != -1 else []

    def handle_service_msg(self,
                           sender: int,
                           msg: ServiceMsg) -> list[Message]:
        if isinstance(msg, RewardMsg):
            new_estim, prev_state = self.receive_reward(msg)
            self.memory.add((prev_state, sender, -new_estim))
            self.__replay()
            return []
        else:
            return super().handle_service_msg(sender, msg)

    def network_changed(self):
        self.node_enc.fit(self.router_graph)

    def __predict(self, states):
        self.brain.eval()
        #print(states)
        output = self.brain(*states)
        return output.clone().detach().numpy()

    # Дообучить модель
    def __train(self, states, targets):
        #print(states)
        self.brain.train()
        self.optimizer.zero_grad()

        output = self.brain(*states)
        loss = self.loss_func(output, targets)
        loss.backward()

        self.optimizer.step()

        return float(loss)

    # Получить состояния для всех соседей
    def __get_nbr_states(self, dst_node):
        cur_emb = self.node_enc.encode(self.id)
        dst_emb = self.node_enc.encode(dst_node)

        others = []
        for inp in self.addit_inputs:
            others.append([])

            if inp['name'] == 'amatrix':
                inp['data'] = nx.to_numpy_array(
                        self.router_graph,
                        weight=None,  # не указывать веса
                        nodelist=sorted(self.router_graph)).flatten()
            else:
                raise Exception('Unknown additional input: ' + inp['name'])

        cur_embs, dst_embs, nbr_embs = [], [], []
        for nbr in sorted(self.out_nbrs):
            cur_embs.append(cur_emb)
            dst_embs.append(dst_emb)

            for i, inp in enumerate(self.addit_inputs):
                others[i].append(inp['data'])

            nbr_embs.append(self.node_enc.encode(nbr))

        states = [np.array(cur_embs),
                  np.array(dst_embs),
                  np.array(nbr_embs),
                  *list(map(np.array, others))]
        return states

    # Обучиться на вознаграждениях
    def __replay(self):
        # Получить batch_size случайных элементов из памяти
        states = []
        estims = []
        for i, (state, sender, new_estim) in \
                self.memory.sample(self.batch_size):
            states.append(state)
            estims.append([new_estim])

        self.__train(stack_rows(states),
                     tch.tensor(estims, dtype=torch.float))
