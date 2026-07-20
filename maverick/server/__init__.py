"""Public API of the server package -- the top of the maverick import stack; every domain and platform module may be imported here, nothing imports back."""

from maverick.server.app import main
from maverick.server.assembly import build_server

__all__ = ["build_server", "main"]
