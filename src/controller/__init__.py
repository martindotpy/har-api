"""App controllers."""

from controller.har_controller import har_router
from controller.health_controller import health_router
from controller.notebook_controller import notebook_router

routers = [health_router, notebook_router, har_router]

__all__ = ["routers"]
