from .base import *
from .qrouter import *
from .linkstate import *


class UnsupportedRouterType(Exception):
    pass

__network_router_classes = {
    'simple_q': QRouter,
    'link_state': LinkStateRouter,
}

def get_router_class(router_type: str):
    try:
        return __network_router_classes[router_type]
    except KeyError:
        raise UnsupportedRouterType(router_type)
