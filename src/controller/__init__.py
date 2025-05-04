"""App controllers."""

from controller.health_controller import health_router

routers = [health_router]

__all__ = ["routers"]
