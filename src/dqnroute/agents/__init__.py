from .base import *
from .qrouter import *
from .linkstate import *
from .dqn import *


class UnsupportedRouterType(Exception):
    pass

__network_router_clses = {
    'q': QRouter,
    'link_state': LinkStateRouter,
    'dqn': DQNRouter,
    'dqn_le': DQNRouter
}

def get_router_class(router_type: str):
    try:
        return __network_router_clses[router_type]
    except KeyError:
        raise UnsupportedRouterType(router_type)
