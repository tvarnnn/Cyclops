"""What the Lab can offer, as JSON, before anything has run.

`IOS-to-Tower.md` 2.1 is the whole requirement and it is deliberately
small:

    "Per experiment, iOS needs: an opaque `id`, a `name`, and optionally
     a `summary`. Nothing else, and nothing is parsed."

and the reason it is small:

    "`docs/modules/EXPERIMENTAL-CV.md` lists nineteen candidates and calls
     the list 'intentionally broad', so any subset hardcoded on the phone
     would be the app asserting that those specific experiments exist."

So this module exists to make the phone's list come from here. Everything
past the three required fields is additive and ignorable -- `provenance`,
`backend`, `headline_label` and the rest are here because an operator
choosing an experiment from a list wants to know which one needs a GPU and
which one is a guess, and because a client that ignores them is unharmed.

Derived from `tower.experiments._REGISTRY` rather than hand-maintained
beside it. A second list would be a second place to forget.
"""

from tower.experiments import EXPERIMENTS, experiment_metadata


def descriptor(experiment_id: str) -> dict:
    """One experiment, as it appears on the wire.

    Raises `KeyError` for an unregistered id, which is the caller's
    problem to turn into a refusal -- this module does not know what a
    client asked for.
    """
    metadata = experiment_metadata(experiment_id)
    return {
        # The three iOS actually reads. `id` is opaque and compared for
        # equality; `name` and `summary` are displayed verbatim.
        "id": experiment_id,
        "name": metadata.name,
        "summary": metadata.summary,
        # Everything below is additive. A client that ignores all of it
        # still has a working experiment picker.
        #
        # Whether the numbers this produces are measurements or model
        # output. Not a hint: iOS makes provenance a REQUIRED field on
        # every metric, and this is where the answer comes from.
        "provenance": metadata.provenance,
        # What it will call its headline number, and in what unit. `null`
        # unit means the quantity genuinely has none and is rendered
        # bare -- never that nobody got around to it.
        "headline_label": metadata.headline_label,
        "headline_unit": metadata.headline_unit,
        # Carries state across frames, so its answer depends on what came
        # before it. Worth knowing before starting one: a stateful
        # experiment's first frame is not like its hundredth.
        "stateful": metadata.stateful,
        # Needs the optional [ml] extra -- torch, weights, possibly a
        # download. A start may take a hundred times longer than a cheap
        # experiment's, which is why it is declared rather than
        # discovered by waiting.
        "requires_model": metadata.requires_model,
        "backend": metadata.backend,
        # The metric that is a count of things found in a frame, if this
        # experiment produces one. iOS renders it as an annotation count,
        # where 0 means "found nothing" and absent means "did not say".
        "annotation_metric": metadata.annotation_metric,
        # What KIND of picture this experiment can draw, or `null` for one
        # that draws none. Here rather than only on the running run so
        # that a phone can say "this one has a live view" in the picker,
        # before anybody commits to a two-minute model load to find out.
        #
        # `null` is a real answer and is not the same as "the preview is
        # unavailable": `baseline` will never have a picture, by design,
        # because it is the control every other experiment's cost is
        # measured against and a picture would double what it costs.
        "preview_kind": metadata.preview_kind,
    }


def catalog() -> list[dict]:
    """Every registered experiment, in a stable order.

    Sorted by id, not by registration order. A list whose order changes
    between two Towers -- or between two runs of one Tower, once dict
    ordering stops being incidental -- would make the payload's revision
    hash change with nothing behind it, and would move a row under
    somebody's finger.
    """
    return [descriptor(experiment_id) for experiment_id in sorted(EXPERIMENTS)]


def is_registered(experiment_id: str) -> bool:
    return experiment_id in EXPERIMENTS
