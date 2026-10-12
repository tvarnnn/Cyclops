"""FOW v1.2 photo floor plan routes (registered by main.py only when
TOWER_WORLD_PHOTO_FLOORPLAN is exactly `on`).

The handlers live in the result-channel adapter
`tower.results.world_builder_photo_floorplan`; this module only re-exports its
router, as `routes/guidance_imagery.py` does, so the web process's route layer never
imports the cartridge itself.
"""

from tower.results.world_builder_photo_floorplan import router

__all__ = ["router"]
