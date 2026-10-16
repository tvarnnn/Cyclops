"""Which COCO classes this cartridge may remember, and why each one.

THE MEASUREMENT THAT DECIDED THIS FILE.

A score histogram describes the DETECTOR'S OPINION OF ITSELF. It cannot
say whether a label is correct, because `data/captures/` carries no
annotation. So the crops were looked at. Every detection
`ssdlite320_mobilenet_v3_large` makes over all 18,821 real frames was
dumped (`scripts/research/object_memory_corpus_dump.py`), grouped into
temporally contiguous SIGHTINGS, and the strongest frame of each sighting
was cropped onto a contact sheet
(`scripts/research/sighting_contact_sheet.py`) and read by eye. The
per-class results are in the table below.

Two findings govern everything here.

**1. Score does not order correctness ACROSS classes.** A ceiling fan in
this home is detected as `airplane` at 0.99 and as `scissors` at 0.93 --
seven and four sightings respectively, none of them an aeroplane or a
pair of scissors. A white door is `refrigerator` at 0.95. A phone held in
a hand is `chair` at 0.94. A laptop keyboard is `remote` at 0.87. Every
one of those is above the 0.5 bar this cartridge uses, and above the
0.813 median score of `laptop`.

**2. The classes a person actually loses are the ones COCO cannot
name.** Keys, a wallet and a pair of glasses have no COCO class at all.
The ones that do exist -- `remote`, `backpack`, `handbag`, `book`,
`bottle` -- produced 8, 1, 0, 1 and 2 sightings across the whole corpus,
and inspection found `backpack` to be a closet of hanging clothes and
`book` to be a laptop screen.

Together those say something specific: **widening the whitelist over this
detector alone would fill a wearer's memory with ceiling fans.** The
class list is therefore not a list of what may be remembered. It is a
list of what may be remembered WITHOUT a second opinion, plus a list of
what may be remembered WITH one -- and the second opinion is a separate,
optional stage that a Tower without a semantic model simply does not
have, in which case those classes are not remembered at all.

THE PRIVACY DECISION IS NOT IN THE TABLE.

`person` is EXCLUDED, and the exclusion is not a tier -- it is a
different kind of entry, in a different constant, checked first, and it
is the one thing no model may overturn. Whether Object Memory may
persist a record per detected bystander is a genuinely open ruling. The
corpus reframes it without answering it: `person` boxes on this footage
have a median area of 38.7% of the frame at the 0.5 threshold this file
uses (35.4% if every detection down to 0.15 is counted) and are usually
the wearer's own torso seen while looking down, so a `person` record
here would mostly be the wearer -- simultaneously less sensitive than
feared and far less useful than hoped. Real bystanders will appear eventually and the ruling
will still be needed. Leaving `person` out is what lets this cartridge
ship without it, and adding it back is not a tuning change.

WHAT THIS CORPUS CANNOT SAY.

34 captures from ONE home, overwhelmingly one activity: a person using a
laptop in a bedroom. `laptop` at 24 correct crops out of 24 is a strong
statement about this laptop in this room, and a weak one about laptops.
No kitchen drawer, no car, no office, no bystander, no set of keys was
ever recorded. Every figure here is a lower bound on how wrong a class
can be, never an upper bound.
"""

from dataclasses import dataclass

# -- the tiers ---------------------------------------------------------
#
# A tier is a decision about EVIDENCE, not about importance. What a class
# is worth remembering is a separate question, and the two come apart:
# `bed` is detected reliably and is not a thing anybody looks for, and
# `remote` is the opposite.

# Remembered on the detector's word alone. Reserved for classes whose
# labels were checked by eye and found right, in this corpus.
REMEMBERED = "remembered"

# Remembered ONLY if a second opinion agrees. These are classes worth
# finding whose labels this detector gets wrong often enough that the
# memory would be worse than empty. With no verifier configured, nothing
# in this tier is ever written, so turning a verifier off is safe rather
# than silently lossy.
VERIFY = "verify"

# Detected, often correctly, and not a memory. Furniture and fittings
# say WHERE somebody was, which is a live question about the current
# scene and belongs to Scene Understanding -- a cartridge that already
# exists and already holds it, without persisting anything. Duplicating
# it here would put a permanent record of a wearer's furniture on disk
# to answer a question another module answers for free.
CONTEXT = "context"

# Everything else. Not an assertion that the class is absent; an
# assertion that this cartridge has no evidence it can read one
# correctly and no reason to try.
IGNORED = "ignored"

TIERS = (REMEMBERED, VERIFY, CONTEXT, IGNORED)


# -- the one entry that is not a tier ----------------------------------
#
# Checked before the table, and unreachable by any model. A verifier may
# demote; it may never promote a name into this set's complement.
EXCLUDED_CLASSES = frozenset({"person"})


@dataclass(frozen=True)
class ClassEvidence:
    """What was measured about one class, and what follows from it.

    `sightings` is a count over the whole 18,821-frame corpus at score
    >= 0.5 with at least 3 frames, and it regenerates exactly from
    `scripts/research/object_memory_corpus_dump.py`.

    `inspected` and `correct` are a HUMAN READING, and their provenance
    is weaker in a way worth stating. They come from one pass over the
    contact sheets `scripts/research/sighting_contact_sheet.py` renders,
    read strongest-first. The sheets regenerate; the per-tile verdicts
    were only recorded individually for the classes that went into
    `scripts/research/open_vocab_verifier_bench.py`'s `GOLDEN` dict, so
    for `bed` (24 read, 20 right) and `chair` (6 read, 5 right) the
    counts here are the only record and cannot be re-derived without
    someone looking again.

    They are carried as data rather than as prose so a later corpus can
    be compared against them, and so a reviewer can see the sample size
    behind a tier instead of trusting the tier.
    """

    tier: str
    sightings: int
    inspected: int
    correct: int
    note: str

    @property
    def precision(self) -> float | None:
        """Fraction of inspected crops whose label was right, or None.

        None when nothing was inspected. Never 0.0 for "not measured" --
        that would read as "always wrong", which is a much stronger
        claim than "unknown".
        """
        if self.inspected == 0:
            return None
        return self.correct / self.inspected


# The table. Ordered by tier, then by sighting count.
#
# Classes absent from this table are IGNORED, which is the default and
# is the safe direction: a COCO class nobody has looked at is a class
# this cartridge has no evidence about.
CLASS_EVIDENCE: dict[str, ClassEvidence] = {
    # -- remembered on the detector's word --------------------------
    "laptop": ClassEvidence(
        REMEMBERED,
        sightings=78,
        inspected=24,
        correct=24,
        note=(
            "24 of 24 strongest crops were laptops. Median best score "
            "0.963 across 32 of 34 captures. The strongest class in the "
            "corpus by a wide margin, and also the most over-represented: "
            "the wearer was using one in most recordings."
        ),
    ),
    "cell phone": ClassEvidence(
        REMEMBERED,
        sightings=80,
        inspected=24,
        correct=24,
        note=(
            "24 of 24 strongest crops were phones. Twenty-two of the 24 "
            "round to 1.00 at two decimals; none is exactly 1.00 and the "
            "highest is 0.9991. Median best score 0.979 across 27 "
            "captures. Small boxes -- a median 8.7% of the frame at the "
            "0.5 threshold this table uses -- and still reliable, which "
            "is unusual here: a lit screen is a strong, distinctive "
            "signal, and it is the one small object this detector is "
            "good at."
        ),
    ),
    # -- worth finding, and not trustworthy without a second opinion --
    #
    # Every entry below is a class the assistive-memory literature or the
    # module brief names as something people look for. None of them is
    # admitted on the detector's word.
    "remote": ClassEvidence(
        VERIFY,
        sightings=8,
        inspected=8,
        correct=3,
        note=(
            "The class this cartridge most needs and least has. The three "
            "highest-scoring sightings (0.87, 0.77, 0.71) are all laptop "
            "keyboards; one more is a phone. Three appear to be a real "
            "remote, all below 0.68. Score orders these EXACTLY WRONG."
        ),
    ),
    "mouse": ClassEvidence(
        VERIFY,
        sightings=4,
        inspected=4,
        correct=3,
        note=(
            "Three crops are one red gaming mouse; the fourth is an "
            "AirPods case. Boxes are ~3% of frame, at the detector's "
            "measured size floor (recall 0.080 for boxes of 2-5% of frame)."
        ),
    ),
    "cup": ClassEvidence(
        VERIFY,
        sightings=3,
        inspected=3,
        correct=3,
        note="Three crops, one drink cup, all correct. Sample far too small to promote.",
    ),
    "bottle": ClassEvidence(
        VERIFY,
        sightings=2,
        inspected=2,
        correct=2,
        note=(
            "Both correct and both ~3% of frame. 922 detections at 0.15 "
            "collapse to 25 at 0.4 and none at 0.7: this is what the size "
            "floor looks like from the other side."
        ),
    ),
    "keyboard": ClassEvidence(
        VERIFY,
        sightings=24,
        inspected=0,
        correct=0,
        note=(
            "Not inspected. Almost certainly co-fires with `laptop` -- 24 "
            "sightings in 12 captures, all of them captures where a laptop "
            "was in view -- so its records would mostly duplicate one that "
            "already exists."
        ),
    ),
    "backpack": ClassEvidence(
        VERIFY,
        sightings=1,
        inspected=1,
        correct=0,
        note=(
            "The single sighting is a closet of hanging clothes, at 0.66. "
            "A named target of the module brief with zero correct "
            "instances in the corpus."
        ),
    ),
    "handbag": ClassEvidence(
        VERIFY,
        sightings=0,
        inspected=0,
        correct=0,
        note="No sighting reached 0.5 across the whole corpus.",
    ),
    "suitcase": ClassEvidence(
        VERIFY,
        sightings=5,
        inspected=5,
        correct=0,
        note=(
            "Four of five crops show a real bag being carried -- but it is "
            "a backpack, not a suitcase. The OBJECT is right and the NAME "
            "is wrong, which is the failure a class whitelist cannot "
            "represent and a verifier can."
        ),
    ),
    "book": ClassEvidence(
        VERIFY,
        sightings=1,
        inspected=1,
        correct=0,
        note="The single sighting is a laptop screen.",
    ),
    "umbrella": ClassEvidence(
        VERIFY,
        sightings=0,
        inspected=0,
        correct=0,
        note="No sighting reached 0.5. Listed because it is a thing people leave behind.",
    ),
    "scissors": ClassEvidence(
        VERIFY,
        sightings=4,
        inspected=4,
        correct=0,
        note="All four sightings are the same ceiling fan, the strongest at 0.93.",
    ),
    "toothbrush": ClassEvidence(
        VERIFY,
        sightings=1,
        inspected=1,
        correct=0,
        note="The single sighting is a boxed tube of toothpaste on a counter.",
    ),
    # -- context: real, and Scene Understanding's question ------------
    "bed": ClassEvidence(
        CONTEXT,
        sightings=66,
        inspected=24,
        correct=20,
        note=(
            "Mostly correct -- this corpus is largely one person on one "
            "bed. Nobody has ever asked where they left their bed."
        ),
    ),
    "tv": ClassEvidence(CONTEXT, 33, 0, 0, "Not inspected. Fixed fitting."),
    "couch": ClassEvidence(CONTEXT, 19, 0, 0, "Not inspected. Fixed fitting."),
    "chair": ClassEvidence(
        CONTEXT,
        sightings=11,
        inspected=6,
        correct=5,
        note="Five real chairs; the sixth is a phone held in a hand, at 0.94.",
    ),
    "sink": ClassEvidence(CONTEXT, 10, 0, 0, "Not inspected. Fixed fitting."),
    "toilet": ClassEvidence(CONTEXT, 9, 0, 0, "Not inspected. Fixed fitting."),
    "refrigerator": ClassEvidence(
        CONTEXT,
        sightings=7,
        inspected=6,
        correct=0,
        note=(
            "All six inspected crops are a white interior door with light "
            "switches, the strongest at 0.95. Kept in CONTEXT rather than "
            "IGNORED only so the count is not lost; nothing here is "
            "written either way."
        ),
    ),
    "dining table": ClassEvidence(
        CONTEXT,
        sightings=0,
        inspected=0,
        correct=0,
        note=(
            "422 detections above 0.15, four above 0.4, and NONE above "
            "the 0.5 this cartridge requires. It is in this table so the "
            "absence is recorded rather than looked up again."
        ),
    ),
    "microwave": ClassEvidence(
        CONTEXT,
        sightings=2,
        inspected=2,
        correct=0,
        note="Both sightings are a monitor showing a bright logo.",
    ),
    "oven": ClassEvidence(CONTEXT, 1, 0, 0, "Not inspected."),
    "potted plant": ClassEvidence(CONTEXT, 0, 0, 0, "No sighting reached 0.5."),
    # -- ignored, and named so the reason survives --------------------
    "airplane": ClassEvidence(
        IGNORED,
        sightings=7,
        inspected=7,
        correct=0,
        note=(
            "Seven sightings, all the same ceiling fan, the strongest at "
            "0.99. The single clearest demonstration in this repository "
            "that a detector score is not a probability of correctness."
        ),
    ),
    "tie": ClassEvidence(
        IGNORED,
        sightings=2,
        inspected=2,
        correct=0,
        note="A door frame and a set of window blinds.",
    ),
    "cat": ClassEvidence(
        IGNORED,
        sightings=14,
        inspected=0,
        correct=0,
        note="Plausibly a real pet. Not a thing anybody looks for.",
    ),
    "dog": ClassEvidence(IGNORED, 9, 0, 0, "As `cat`."),
}


# What a wearer might plausibly ask this cartridge to find, as opposed to
# what it happens to detect. Drawn from the module brief's own examples
# ("Where did I last see my keys?", "Have I seen my backpack today?",
# "When was my charger last visible?", "What room did I leave my water
# bottle in?") and from the assistive-memory literature, which converges
# on the same handful: keys, wallet, phone, glasses, remote, medication.
#
# Keys, a wallet, glasses, a charger and medication have no COCO class
# (the literature's six are keys, wallet, phone, glasses, remote,
# medication; `phone` and `remote` do have COCO classes, `charger` is
# from the module brief). This detector cannot name them at any
# threshold, which is a ceiling on what a COCO-only cartridge can ever
# be for. Recorded here so the gap is visible from the code that has it.
WORTH_FINDING_WITHOUT_A_COCO_CLASS = ("keys", "wallet", "glasses", "charger", "medication")


# --- what a verifier is asked, and what it is asked to choose between ---
#
# A COCO class name is not always the phrase a language-conditioned model
# understands best. `mouse` alone is an animal.
PROMPT_FOR = {
    "mouse": "computer mouse",
    "keyboard": "computer keyboard",
    "remote": "remote control",
    "cup": "drinking cup",
    "tv": "television screen",
    "microwave": "microwave oven",
    "tie": "necktie",
}

# Every alternative a verifier gets to choose from, every time.
#
# Fixed rather than per-crop, and that is the whole reason a verifier
# works at all. Asked "is this a remote?" with nothing else on offer, a
# model says yes -- there is nothing else to say. Given the alternatives,
# and specifically the ones this detector actually confuses, it says
# "computer keyboard" and the sighting is refused.
#
# Measured on 94 human-labelled crops from the real corpus: with this
# vocabulary `owlv2-base-patch16-ensemble` accepts ~93% of correct labels
# and rejects ~94% of wrong ones -- two figures rather than three, because
# 81% of the positives in that set are two block assertions and a third
# significant figure would be more precision than the labels carry.
#
# `human hand` is here as a DISTRACTOR and nothing else. A crop of
# something held in a hand is the commonest confuser in this corpus, and
# a model allowed to say so rejects it. No score against it is ever
# stored, no record is ever created from it, and `person` is deliberately
# NOT in the list -- the exclusion above is not a thing a prompt list
# should be able to work around.
VERIFIER_DISTRACTORS = (
    "ceiling fan",
    "airplane",
    "door",
    "door frame",
    "wall",
    "window blinds",
    "television screen",
    "computer monitor",
    "clothes on hangers",
    "human hand",
    "bed",
    "couch",
    "chair",
    "sink",
    "toilet",
    "refrigerator",
    "microwave oven",
)


# --- parts, and the wholes that make them redundant ---------------------
#
# A class whose sighting is usually a PART of another class's sighting.
# Measured rather than assumed: `keyboard` produced 24 sightings across
# 12 captures over the real corpus, and every one of those captures also
# had a laptop in view. A replay of the validated capture with the
# verifier on and no such rule wrote two `keyboard` records, and
# inspection showed one of them to be the keyboard of a laptop that
# already had its own record: the same object, remembered twice under
# two names. With the rule, the laptop's own keyboard is not written;
# the standalone keyboard is.
#
# Suppression is CONCURRENT, not blanket, and the distinction is the
# whole reason this is a table rather than a tier change. Latching it
# onto the sighting would let one frame of a laptop silence a keyboard
# for the rest of an unbounded sighting. The other `keyboard` record in
# that replay is a lit mechanical keyboard at a desk with no laptop
# anywhere near it, which is a real object somebody could genuinely go
# looking for. A blanket rule would lose it; a rule that only fires
# while the whole is ALSO in view keeps it.
PART_OF = {
    "keyboard": ("laptop",),
}


def wholes_of(object_class: str) -> tuple[str, ...]:
    return PART_OF.get(object_class, ())


def prompt_for(object_class: str) -> str:
    return PROMPT_FOR.get(object_class, object_class)


def verifier_vocabulary() -> tuple[str, ...]:
    """Every phrase a verifier may rank, in a stable order.

    Persistable classes first so the list reads as "the things this
    cartridge could remember, and everything they get confused with".
    Duplicates are removed while keeping the first occurrence, because a
    repeated prompt is a repeated query and costs latency for nothing.
    """
    seen = {}
    for name in PERSISTABLE_CLASSES:
        seen.setdefault(prompt_for(name), None)
    for name in VERIFIER_DISTRACTORS:
        seen.setdefault(name, None)
    return tuple(seen)


def tier_of(object_class: str) -> str:
    """Which tier a class falls in. Unknown classes are IGNORED.

    `person` -- and anything else in EXCLUDED_CLASSES -- resolves to
    IGNORED here too, so a caller that only consults tiers still cannot
    persist one. The separate constant exists because the REASON differs
    and a reader must be able to tell "no evidence" from "not allowed".
    """
    if object_class in EXCLUDED_CLASSES:
        return IGNORED
    evidence = CLASS_EVIDENCE.get(object_class)
    return evidence.tier if evidence is not None else IGNORED


def is_excluded(object_class: str) -> bool:
    return object_class in EXCLUDED_CLASSES


def classes_in(tier: str) -> tuple[str, ...]:
    """Every class in one tier, in DECLARATION order, not alphabetical.

    The order reaches the wire: `recorded_classes` on every payload is
    this list, and a client that has already shipped compares it. Sorting
    would have reordered `("laptop", "cell phone")` -- the exact list the
    iOS decoder was written against -- for no reason but tidiness. The
    table above is ordered strongest-evidence-first within each tier,
    which is also the order a reader wants.
    """
    return tuple(
        name
        for name, evidence in CLASS_EVIDENCE.items()
        if evidence.tier == tier and name not in EXCLUDED_CLASSES
    )


# Every class that may ever reach the store, under any configuration.
#
# This is the STORE'S allowlist, not the producer's decision: a `verify`
# class is only written when a verifier agreed, but the store cannot see
# a verifier and must not have to. It enforces the outer bound -- nothing
# outside these names, ever, from any caller -- and the producer enforces
# the inner one.
PERSISTABLE_CLASSES = classes_in(REMEMBERED) + classes_in(VERIFY)

# Alias of `PERSISTABLE_CLASSES`: the store's default allowlist and
# `object_query`'s validation list.
PERSISTED_CLASSES = PERSISTABLE_CLASSES
