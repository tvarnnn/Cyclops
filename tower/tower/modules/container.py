import asyncio
import logging
from typing import Any

from tower.modules.base import (
    FrameProcessingError,
    FrameSkippedError,
    Module,
    ModuleDescriptor,
    ModuleState,
    ModuleUnavailableError,
)

logger = logging.getLogger(__name__)

LIFECYCLE_TIMEOUT_S = 10.0

# Load gets its own bound, and it is deliberately two orders of magnitude
# looser than the others. The number comes from what a first load
# actually has to do, measured on this host's caches rather than chosen
# for roundness:
#
#   * depth (MiDaS-small) downloads 85.8 MB of weights plus a 33.2 MB
#     tf_efficientnet_lite3 backbone -- 119 MB -- and clones two
#     torch.hub repositories before any of it;
#   * object_detection / the shared SSDLite detector downloads 14.1 MB.
#
# 119 MB inside the 10 s general bound demands ~95 Mbit/s sustained from
# the first byte, including TLS setup and GitHub redirects. No ordinary
# link guarantees that, so enforcing 10 s on load would not have fixed a
# hang -- it would have converted "hangs once, then works forever" into
# "fails every first run until somebody pre-warms the cache", which is
# the worse bug. 120 s covers the same download at ~8 Mbit/s.
#
# It is still a bound, not a surrender. A WARM load of the same models
# measures 1.8 s (depth, CUDA, cached) and 0.16 s (SSDLite, CPU, cached)
# on this host, so 120 s is roughly 65x the real cost: a genuinely stuck
# load -- a stalled download, an unreachable host -- is still caught, in
# bounded time, by a Tower that then reports FAILED instead of hanging.
#
# start/stop/unload keep the tight bound. None of them touch the network.
LOAD_TIMEOUT_S = 120.0


class ModuleContainer:
    """Drives the single hardcoded V0.8 module through its full lifecycle.

    Registry of one: this container holds exactly one Module instance,
    constructed by the caller. No discovery, no swapping.
    """

    def __init__(
        self,
        module: Module,
        lifecycle_timeout_s: float | None = None,
        load_timeout_s: float | None = None,
    ) -> None:
        self._module = module
        # The question both defaults answer is "did the caller specify a
        # bound?", not "what number did they pick?". A caller that
        # narrows the container to 50 ms is not asking for 50 ms
        # everywhere EXCEPT load, which may take two minutes -- it is
        # asking for a tightly bounded container, and load is part of it.
        #
        # This used to compare `lifecycle_timeout_s >= LIFECYCLE_TIMEOUT_S`
        # instead, which cannot tell a caller who passed 10.0 on purpose
        # from one who passed nothing. That made the load bound
        # discontinuous and non-monotonic: 9.999 gave a 9.999 s load
        # bound and 10.0 gave 120 s (a 12,000x swing on a 1 ms change),
        # while an explicit 300 -- a caller widening everything -- got a
        # TIGHTER 120 s load bound than they asked for. A `None` sentinel
        # asks the real question, so explicit 9.999 stays 9.999, explicit
        # 300 stays 300, and only an unspecified bound gets the default.
        if load_timeout_s is None:
            load_timeout_s = (
                LOAD_TIMEOUT_S if lifecycle_timeout_s is None else lifecycle_timeout_s
            )
        if lifecycle_timeout_s is None:
            lifecycle_timeout_s = LIFECYCLE_TIMEOUT_S
        self._lifecycle_timeout_s = lifecycle_timeout_s
        self._load_timeout_s = load_timeout_s

    @property
    def state(self) -> ModuleState:
        return self._module.state

    @property
    def descriptor(self) -> "ModuleDescriptor":
        return self._module.descriptor

    async def load_and_start(self) -> None:
        try:
            await asyncio.wait_for(
                self._module.load(), timeout=self._load_timeout_s
            )
            await asyncio.wait_for(
                self._module.start(), timeout=self._lifecycle_timeout_s
            )
        except Exception:
            # A load timeout ABANDONS the loader; it cannot stop it. The
            # module is marked FAILED here and released immediately, while
            # the loader may still be seconds from finishing. Whatever it
            # finishes with must not be installed -- see
            # `tower/loading.py`, which is what keeps that promise.
            logger.exception(
                "module %s failed to load/start; marking FAILED",
                self._module.descriptor.id,
            )
            self._module.mark_failed()

    async def shutdown(self) -> None:
        if self._module.state == ModuleState.ACTIVE:
            try:
                await asyncio.wait_for(
                    self._module.stop(), timeout=self._lifecycle_timeout_s
                )
            except Exception:
                logger.exception(
                    "module %s failed to stop; marking FAILED",
                    self._module.descriptor.id,
                )
                self._module.mark_failed()
                return
        if self._module.state == ModuleState.READY:
            try:
                await asyncio.wait_for(
                    self._module.unload(), timeout=self._lifecycle_timeout_s
                )
            except Exception:
                logger.exception(
                    "module %s failed to unload; marking FAILED",
                    self._module.descriptor.id,
                )
                self._module.mark_failed()

    def process(self, raw_bytes: bytes) -> Any:
        """Run one raw frame through the single CV module, if it is ACTIVE.

        A module in any other state raises `ModuleUnavailableError` naming that
        state. A `FrameProcessingError` from the module is frame-level: the module
        stays ACTIVE and the error becomes `FrameSkippedError`, carrying the
        module's own `reason` and wording when it named one. Any other exception is
        a crash: the module is marked FAILED and `ModuleUnavailableError` is raised.
        Returns the module's result unchanged.
        """
        if self._module.state != ModuleState.ACTIVE:
            # state.value, not the enum: this string reaches the WS client
            # in a frame_error message, so it should carry the protocol's
            # own vocabulary ("failed") rather than a Python repr
            # ("ModuleState.FAILED").
            raise ModuleUnavailableError(
                f"module {self._module.descriptor.id} is "
                f"{self._module.state.value}, not ACTIVE"
            )
        try:
            return self._module.process(raw_bytes)
        except FrameProcessingError as exc:
            logger.warning(
                "module %s: frame-level failure, module stays ACTIVE: %s",
                self._module.descriptor.id,
                exc,
            )
            # A module that named a REASON also owns the WORDING: it knows
            # what it refused and what a person should do about it, and
            # "could not process this frame" would throw that away. A
            # module that named nothing keeps the generic sentence, which
            # is what every existing caller and test expects.
            reason = getattr(exc, "reason", None)
            raise FrameSkippedError(
                str(exc)
                if reason is not None
                else (
                    f"module {self._module.descriptor.id} could not process "
                    "this frame"
                ),
                reason=reason,
            ) from exc
        except Exception as exc:
            logger.exception(
                "module %s raised during process(); marking FAILED",
                self._module.descriptor.id,
            )
            self._module.mark_failed()
            raise ModuleUnavailableError(
                f"module {self._module.descriptor.id} failed while processing"
            ) from exc
