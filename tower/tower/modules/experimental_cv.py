from tower.cv_lab import CVLab
from tower.experiments import ExperimentResult, ExperimentSettings
from tower.modules.base import Module, ModuleDataBehavior, ModuleDescriptor

DESCRIPTOR = ModuleDescriptor(
    id="experimental-cv",
    name="Experimental CV Lab",
    version="0.3.0",
    data_behavior=ModuleDataBehavior(
        persists_data=False,
        retains_raw_imagery=False,
        retention="none",
        supports_purge=False,
        transmits_externally=False,
    ),
)


class ExperimentalCVModule(Module):
    """The one Lab slot in the container. What is IN it is `CVLab`'s job.

    This class used to hold the experiment itself. It no longer does, and
    the reason is not tidiness: an experiment chosen at runtime has a
    lifecycle of its own -- armed, paused, swapped, failed and retried --
    and none of those are module transitions. Folding them into
    `ModuleState` would have meant either a module that goes READY and
    ACTIVE several times a minute, or `mark_failed()` (which is TERMINAL)
    firing because a weight download timed out.

    So the split is: the container drives this module exactly as it always
    did, once, at startup; and `CVLab` drives what the module is holding.
    A failed experiment leaves the Lab recoverable and the module ACTIVE,
    which is what makes "try the next experiment" a request rather than a
    restart.

    `data_behavior` still declares no persistence and no retained imagery,
    and that must stay true: an experiment that wanted to persist would
    make this descriptor a lie, and the descriptor is what the privacy
    policy is enforced against.

    **The live preview does not make it a lie, and the reasoning is worth
    writing down rather than assuming.** `persists_data` is unchanged
    because nothing reaches a disk: one image lives in memory, is replaced
    by the next, and is gone when the run stops.
    `retains_raw_imagery` is unchanged because nothing retained is raw --
    an experiment holds a Canny edge map, a depth array or a set of
    boxes, never the frame, and `experiments.scene_structure` is where
    that guarantee is actually implemented. Neither field would survive a
    design that served a filtered photograph, which is one of the reasons
    this one does not.
    """

    descriptor = DESCRIPTOR

    def __init__(
        self,
        experiment_name: str,
        settings: ExperimentSettings | None = None,
        experiment=None,
        *,
        connection_count=None,
        preview=None,
    ) -> None:
        super().__init__()
        # Constructed here rather than injected so that every caller which
        # built a module before this change still builds a working one,
        # and so there is exactly one Lab per module -- a second Lab
        # sharing a slot is the "two experiments at once" failure the
        # whole design exists to make impossible.
        self.lab = CVLab(
            experiment_name,
            settings or ExperimentSettings(),
            experiment=experiment,
            connection_count=connection_count,
            # Passed straight through. This class holds the Lab and does
            # not hold policy: `main.py` reads `Settings`, and every
            # existing caller that built a module without one still gets a
            # working Lab with previews on.
            preview=preview,
        )

    async def _do_load(self) -> None:
        # Propagates, deliberately. An unknown name or a failed load of
        # the STARTUP DEFAULT reaches ModuleContainer, which marks the
        # module FAILED -- a typo in TOWER_CV_EXPERIMENT must still be
        # loud, and every existing lifecycle and load-timeout test encodes
        # that. The recoverable path is the interactive one; see
        # `CVLab.start`.
        #
        # The off-thread load, and the reason it must be off-thread, now
        # live in `CVLab.load_initial`. What matters here is only that
        # this still awaits, so the container's load timeout still bounds
        # it.
        await self.lab.load_initial()

    async def _do_start(self) -> None:
        return None

    def _do_process(self, observation: bytes) -> ExperimentResult:
        return self.lab.process(observation)

    async def _do_stop(self) -> None:
        return None

    async def _do_unload(self) -> None:
        self.lab.release("the Lab module was unloaded")

    def _do_release(self) -> None:
        self.lab.release("the Lab module failed and was released")
