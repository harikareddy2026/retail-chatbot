import os
from dotenv import load_dotenv
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Load the .env file so we can read MCP_API_KEY
load_dotenv()

# Read the valid key from .env once at startup
# We cache it here so we don't read the file on every request
VALID_KEY = os.getenv("MCP_API_KEY")

if not VALID_KEY:
    raise ValueError("MCP_API_KEY not found in .env file!")

class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    This runs before every request.
    It checks the X-API-Key header.
    If missing or wrong → returns 401 Unauthorized.
    If correct → lets the request through normally.
    """
    async def dispatch(self, request, call_next):

        # Skip auth check for /health endpoint
        # Docker and load balancers need unauthenticated health checks
        if request.url.path == "/health":
            return await call_next(request)

        # Get the key from the request header
        incoming_key = request.headers.get("X-API-Key", "")

        # Compare with our valid key
        if incoming_key != VALID_KEY:
            # Return 401 Unauthorized - do not proceed to endpoint
            return Response(
                content="Unauthorized - invalid or missing API key",
                status_code=401
            )

        # Key is correct - let the request through
        return await call_next(request)