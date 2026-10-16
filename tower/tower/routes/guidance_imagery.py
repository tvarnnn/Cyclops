"""Guidance imagery routes (always registered by main.py).

The handlers live in the result-channel adapter
`tower.results.world_builder_guidance_imagery`; this module only re-exports
its router, as `routes/photo_floorplan.py` does, so the web process never
imports the cartridge itself.
"""

from tower.results.world_builder_guidance_imagery import router

__all__ = ["router"]
