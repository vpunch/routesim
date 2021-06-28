from .base import *
from .qrouter import *
from .linkstate import *
from .dqn import *


class UnsupportedRouterType(Exception):
    pass

__network_router_classes = {
    'q': QRouter,
    'link_state': LinkStateRouter,
    'dqn': DQNRouter,
    'dqn-le': DQNRouter
}

def get_router_class(router_type: str):
    try:
        return __network_router_classes[router_type]
    except KeyError:
        raise UnsupportedRouterType(router_type)
