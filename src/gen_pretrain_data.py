import os
import argparse

import yaml
import networkx as nx
import pandas as pd

from dqnroute.utils import *

seed = 41


def parse_actions(nodes, pkg_distr):
    random.seed(seed)

    pkg_id = 0
    for item in pkg_distr:
        print(pkg_id, item)

        action = item.get('action', 'send_pkgs')
        if action == 'send_pkgs':
            srcs = item.get('srcs', nodes)
            dsts = item.get('dsts', nodes)
            for _ in range(item['num']):
                # Выбрать отличные узлы отправки и назначения
                src, dst = 0, 0
                while src == dst:
                    src = random.choice(srcs)
                    dst = random.choice(dsts)

                yield 'send_pkg', (pkg_id, src, dst)

                pkg_id += 1
        elif action == 'break_link' or action == 'restore_link':
            yield action, (item['u'], item['v'])
        else:
            raise Exception('Unexpected action: ' + action)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-training dataset generator')
    parser.add_argument('launch', type=str, help='Path to launch file')
    parser.add_argument('output', type=str, help='Path to results .csv')
    args = parser.parse_args()

    # Загрузить настройки
    launch = open(args.launch)
    run_params = yaml.safe_load(launch)
    launch.close()

    settings = run_params['settings']

    # Создать граф сети
    net_graph = nx.Graph()
    for e in run_params['network']:
        u = e['u']
        v = e['v']

        trans_time = settings['pkg_size'] / e['bandwidth'] \
            + settings['router_env']['pkg_proc_delay']
        net_graph.add_edge(u, v, weight=trans_time)

    links_data = {}

    amatrix_cols = get_multi_col('amatrix', len(net_graph)**2)
    data_cols = ['dst', 'src', 'pkg_id', 'nbr'] + amatrix_cols + ['estim']
    df = pd.DataFrame(columns=data_cols)

    def get_amatrix(net_graph):
        # В матрице смежности узлы упорядочены
        # Во время обучения модели эмбеддинги обучаются по матрице смежности,
        # поэтому важно включить информацию о весах
        return list(nx.to_numpy_array(
                net_graph,
                weight='weight',
                nodelist=sorted(net_graph.nodes())).flatten())

    amatrix = get_amatrix(net_graph)

    for action, params in parse_actions(list(net_graph),
                                        settings['pkg_distr']):
        if action == 'send_pkg':
            pkg_id, src_node, dst_node = params
            for cur_node in nx.dijkstra_path(net_graph, src_node, dst_node):
                # По кратчайшему пути

                for nbr_node in list(net_graph.neighbors(cur_node)):
                    # Оценить время доставки через каждого соседа

                    # Получить состояние
                    state = [dst_node, cur_node, pkg_id, nbr_node] + amatrix

                    # Получить целевое значение
                    try:
                        estim = net_graph.edges[cur_node, nbr_node]['weight']
                        estim += nx.dijkstra_path_length(net_graph,
                                                         nbr_node,
                                                         dst_node)
                    except nx.exception.NetworkXNoPath:
                        estim = -INFTY

                    # Записать объект
                    df.loc[len(df)] = state + [-estim]
        elif action == 'break_link':
            u, v = params
            links_data[(u, v)] = net_graph[u][v]
            net_graph.remove_edge(u, v)
            amatrix = get_amatrix(net_graph)
        elif action == 'restore_link':
            u, v = params
            net_graph.add_edge(u, v, **links_data.pop((u, v)))
            amatrix = get_amatrix(net_graph)
        else:
            raise Exception('Unexpected action: ' + action)

    df.to_csv(args.output, mode='w', header=True, index=True)


if __name__ == '__main__':
    main()
