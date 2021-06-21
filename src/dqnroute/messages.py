# Основные виды сообщений:
# Message -- любое сообщение в сети
# ServiceMsg(Message) -- сообщение для настройки маршрутизации
# Package -- пакет, доставка которого исследуется

# Сообщения-контейнеры (имеют атрибут inner_msg):
# DelayedMsg(Message)
# __TransferMsg(Message)

from functools import total_ordering


# Базовый класс для всех сообщений в сети
class Message:
    def __init__(self, **kwargs):
        self.content = kwargs

    def __str__(self):
        return f'{self.__class__.__name__}: {str(self.content)}'

    # Позволить обращаться к содержимому при помощи атрибутов
    def __getattr__(self, name):
        try:
            return self.content[name]
        except KeyError:
            raise AttributeError(name)


# Исключение для неподдерживаемых сообщений
class UnsupportedMsgType(Exception):
    pass


# Сообщение, которое должно быть обработано с задержкой
class DelayedMsg(Message):
    def __init__(self, id: int, delay: float, inner_msg: Message):
        super().__init__(id=id, delay=delay, inner_msg=inner_msg)


# Сообщение для немедленного выполнения отложенного сообщения
class InterruptDelayMsg(Message):
    def __init__(self, delay_id: int):
        super().__init__(delay_id=delay_id)


# Сообщение, которое получают все роутеры, когда сеть построена
class InitMsg(Message):
    def __init__(self, config):
        super().__init__(config=config)


# Сообщение, которое передается между узлами
class __TransferMsg(Message):
    def __init__(self,
                 from_node: int,
                 to_node: int,
                 inner_msg: Message):
        super().__init__(from_node=from_node,
                         to_node=to_node,
                         inner_msg=inner_msg)


# Входящее в узел сообщение
class InMsg(__TransferMsg):
    pass


# Исходящее из узла сообщение
class OutMsg(__TransferMsg):
    pass


class ServiceMsg(Message):
    pass


@total_ordering  # для вывода остальных операций сравнения
class Package:
    def __init__(self, pkg_id, size, dst, start_time, content):
        self.id = pkg_id
        self.size = size
        self.dst = dst
        self.start_time = start_time
        self.content = content

    def __eq__(self, other):
        return self.id == other.id

    def __lt__(self, other):
        return self.id < other.id


# Сообщение, содержащее пакет
class PkgMsg(Message):
    def __init__(self, pkg: Package):
        super().__init__(pkg=pkg)


# Сообщение, которое отправляет роутер, если пакет доставлен
class PkgReceivedMsg(Message):
    def __init__(self, pkg: Package):
        super().__init__(pkg=pkg)


class LinkUpdateMsg(Message):
    def __init__(self, node: int, direction='both', **kwargs):
        super().__init__(node=node, direction=direction, **kwargs)


class AddLinkMsg(LinkUpdateMsg):
    def __init__(self, node: int, direction='both', edge_data={}):
        super().__init__(node, direction, edge_data=edge_data)


class RemoveLinkMsg(LinkUpdateMsg):
    def __init__(self, node: int, direction='both'):
        super().__init__(node, direction)


# Сообщение для расчета награды
class RewardMsg(ServiceMsg):
    def __init__(self, pkg_id: int, estim: float, reward_data):
        super().__init__(pkg_id=pkg_id, estim=estim, reward_data=reward_data)


class StateAnnounMsg(ServiceMsg):
    def __init__(self, node: int, seq: int, state):
        super().__init__(node=node, seq=seq, state=state)
