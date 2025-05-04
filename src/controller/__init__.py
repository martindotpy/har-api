"""App controllers."""

from controller.health_controller import health_router
from controller.notebook_controller import notebook_router

routers = [health_router, notebook_router]

__all__ = ["routers"]
