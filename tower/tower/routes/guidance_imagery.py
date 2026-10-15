"""FOW v1.1 B guidance imagery routes (always registered by main.py).

The handlers live in the result-channel adapter
`tower.results.world_builder_guidance_imagery`; this module only re-exports
its router, as `routes/geometry.py` does for the other World Builder adapters,
so the web process never imports the cartridge itself.
"""

from tower.results.world_builder_guidance_imagery import router

__all__ = ["router"]
