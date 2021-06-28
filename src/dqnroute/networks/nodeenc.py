from warnings import warn

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

# Удалять узлы нельзя, можно добавлять новые


class NodeEnc:
    def __init__(self, dim):
        self.dim = dim

        self.embs = {}

    def fit(self, net_graph):
        # Эмбеддинги можно обучать по матрице смежности
        if isinstance(net_graph, np.ndarray):
            # Во время обучения модели матрица смежности передается в плоском
            # виде
            if net_graph.ndim == 1:
                net_graph = net_graph.reshape(int(len(net_graph)**(1/2)), -1)

            net_graph = nx.from_numpy_array(net_graph, create_using=nx.DiGraph)

        self.fit_(net_graph)

    def encode_(self, net_graph):
        raise NotImplementedError()

    def encode(self, node):
        if node not in self.embs:
            warn(f'Have not embedding for node: {node}')

        return self.embs.get(node, np.zeros(self.dim))


# One-Hot Node Encoder
class OHNodeEnc(NodeEnc):
    def __init__(self, dim):
        super().__init__(dim)

    def fit_(self, net_graph):
        nodes = sorted(net_graph)[:self.dim]
        embs = np.zeros((len(nodes), self.dim))
        embs[range(len(nodes)), nodes] = 1
        self.embs = dict(zip(nodes, embs))


# Laplacian Eigenmap Node Encoder
class LENodeEnc(NodeEnc):
    def __init__(self, dim):
        super().__init__(dim)

    def fit_(self, net_graph):
        if len(net_graph) <= self.dim + 1:
            return

        # Алогоритм не работает с ориентированными графами
        net_graph = net_graph.to_undirected()

        # Эмбеддинги на графах, которые отличаются только весом, должны быть
        # разными. Для этого найдем средний вес, на него разделим каждый вес и
        # домножим эмбеддинги.

        w_total = net_graph.size(weight='weight')
        e_num = net_graph.size(weight=None)
        w_avg = w_total / e_num

        for u, v in net_graph.edges():
            net_graph[u][v]['weight'] /= w_avg

        # Чем больше вес ребра, тем ближе находятся узлы. Нужно сделать
        # наоборот.
        for u, v, w in net_graph.edges.data('weight'):
            net_graph[u][v]['weight'] **= -1

            # Можно использовать распределение Больцмана с низкой
            # температурой, чтобы маленькие веса экспоненциально не
            # увеличивали значение функции.
            #net_graph[u][v]['weight'] = exp(-w)

        # Матрица смежности графа в виде рязряженной матрицы
        # Симметричная, так как граф ненаправленный
        A = nx.to_scipy_sparse_matrix(
                net_graph,
                nodelist=sorted(net_graph),
                weight='weight')

        # Суммировать столбцы (вес каждой вершины)
        diags = np.array(A.sum(axis=0))
        # Диагональная матрица
        D = sp.diags(diags, [0])
        # Матрица Кирхгофа
        L = D - A

        # Начальный вектор должен быть определен, чтобы во время обучения и
        # симуляции получались одинаковые эмбеддинги
        # Вектор из единиц вызывает ошибку в scipy 1.7, в 1.6 возможно будет
        # работать
        v0 = [0.0781944, 0.03992914, 0.0276535, 0.66376005, 0.01232996]
        v0 = [v0[i % len(v0)] for i in range(A.shape[0])]

        # Ly = \Dy
        # Ax = wMx
        _, vectors = sp.linalg.eigsh(
                L,             # симметричная матрица
                k=self.dim+1,  # сколько искать
                M=D,
                which='SM',    # наименьшие значения по модулю (magnitude)
                v0=v0)

        embs = vectors[:, 1:]
        embs *= w_avg

        self.embs = dict(zip(sorted(net_graph), embs))


__node_enc_clses = {
    'oh': OHNodeEnc,
    'le': LENodeEnc
}


def node_encoder_class(name):
    return __node_enc_clses[name]
