import importlib.metadata

# __version__ = importlib.metadata.version("mem0ai")
__version__ = "local"
from mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from mem0.memory.main import AsyncMemory, Memory  # noqa
