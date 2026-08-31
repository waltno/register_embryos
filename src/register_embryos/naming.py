"""Filename contract for HCR ND2 files, and the cohort grouping it implies.

Every ND2 file must be named

    {date}_{id}_{genotype}_{timepoint}_{view}_{magnification}_{ch1}_{ch2}_{ch3}.nd2

for example ``20260410_1.5_wt_12s_dorsal_20X_hand2_tbx5a_wt1a.nd2``.

Channel 0 is always the nuclear stain (DAPI); ``ch1..ch3`` name the gene
channels in acquisition order, so ``{1: "hand2", 2: "tbx5a", 3: "wt1a"}``.

Embryos are processed together when they share
``(genotype, timepoint, view, magnification)`` — that 4-tuple is the *cohort*,
and it is also the name of the cohort's output directory.  The channel identity
is deliberately NOT part of the cohort key: two embryos of the same genotype and
stage imaged the same way belong in one registered space even when the gene
panel differs, which is exactly the wt1a-plus-rotating-partner design.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "FILENAME_SPEC",
    "EmbryoName",
    "CohortKey",
    "parse_embryo_name",
    "format_embryo_name",
    "discover_embryos",
    "group_into_cohorts",
    "plan_renames",
    "apply_renames",
    "undo_renames",
    "MANIFEST_NAME",
]

FILENAME_SPEC = "date_id_genotype_timepoint_view_magnification_channel1_channel2_channel3"
MANIFEST_NAME = "rename_manifest.csv"

#: Tokens recognised as an imaging view, however they are cased.
VIEW_TOKENS = frozenset(
    {"dorsal", "ventral", "lateral", "anterior", "posterior", "transverse", "sagittal"}
)
#: A magnification token is a number followed by x/X, e.g. 10X, 20x, 63X, 1.5X.
MAGNIFICATION_RE = re.compile(r"^\d+(?:\.\d+)?[xX]$")
#: An 8-digit acquisition date, YYYYMMDD.
DATE_RE = re.compile(r"^\d{8}$")
#: A timepoint: somite stage (12s), hours/days post fertilisation (24hpf, 3dpf),
#: or a bare hpf number (24h).
TIMEPOINT_RE = re.compile(r"^\d+(?:\.\d+)?(?:s|somite|somites|hpf|dpf|h|hr|hrs|d)$", re.I)


def _is_view(token: str) -> bool:
    return token.lower() in VIEW_TOKENS


def _is_magnification(token: str) -> bool:
    return bool(MAGNIFICATION_RE.match(token))


def _is_timepoint(token: str) -> bool:
    return bool(TIMEPOINT_RE.match(token))


@dataclass(frozen=True)
class CohortKey:
    """The 4-tuple that decides which embryos are processed together."""

    genotype: str
    timepoint: str
    view: str
    magnification: str

    @property
    def name(self) -> str:
        """Directory name for this cohort's outputs."""
        return f"{self.genotype}_{self.timepoint}_{self.view}_{self.magnification}"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.name


@dataclass(frozen=True)
class EmbryoName:
    """One parsed ND2 filename.

    ``embryo_id`` is the full stem, which is what every downstream table keys on,
    so it stays stable across the whole workflow.
    """

    date: str
    embryo_num: str
    genotype: str
    timepoint: str
    view: str
    magnification: str
    channels: Tuple[str, ...]
    path: Optional[Path] = None
    extra: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def embryo_id(self) -> str:
        return format_embryo_name(self)

    @property
    def cohort(self) -> CohortKey:
        return CohortKey(self.genotype, self.timepoint, self.view, self.magnification)

    @property
    def gene_map(self) -> Dict[int, str]:
        """Channel index -> gene name, starting at channel 1 (channel 0 is nuclei)."""
        return {i: gene for i, gene in enumerate(self.channels, start=1)}

    def with_timepoint(self, timepoint: str) -> "EmbryoName":
        return EmbryoName(
            date=self.date,
            embryo_num=self.embryo_num,
            genotype=self.genotype,
            timepoint=timepoint,
            view=self.view,
            magnification=self.magnification,
            channels=self.channels,
            path=self.path,
            extra=self.extra,
        )


def format_embryo_name(name: EmbryoName) -> str:
    """Render an :class:`EmbryoName` back into the canonical stem."""
    parts = [
        name.date,
        name.embryo_num,
        name.genotype,
        name.timepoint,
        name.view,
        name.magnification,
        *name.channels,
        *name.extra,
    ]
    return "_".join(str(p) for p in parts)


def parse_embryo_name(
    path: str | Path,
    default_timepoint: Optional[str] = None,
    strict: bool = False,
) -> EmbryoName:
    """Parse an ND2 path into its fields.

    The parser is order-tolerant for ``view`` and ``magnification`` because those
    two get swapped by hand often enough to matter -- ``20X_dorsal`` and
    ``dorsal_20X`` both parse, and both normalise to view-then-magnification on
    output.  A missing ``timepoint`` is filled from ``default_timepoint`` (the CLI
    passes one inferred from the containing directory) unless ``strict`` is set.

    Raises:
        ValueError: if the name cannot be parsed, or if ``strict`` and a field
            had to be inferred rather than read.
    """
    path = Path(path)
    tokens = path.stem.split("_")
    if len(tokens) < 6:
        raise ValueError(
            f"{path.name!r}: expected at least 6 underscore-separated fields "
            f"({FILENAME_SPEC}), found {len(tokens)}"
        )

    date = tokens[0]
    if not DATE_RE.match(date):
        raise ValueError(f"{path.name!r}: field 1 {date!r} is not an 8-digit YYYYMMDD date")
    embryo_num = tokens[1]

    # Locate the view and magnification tokens wherever they sit.  Everything
    # before the first of the two (after date/id) is genotype + optional
    # timepoint; everything after the second is the channel list.
    view_idx = next((i for i, t in enumerate(tokens) if i >= 2 and _is_view(t)), None)
    mag_idx = next((i for i, t in enumerate(tokens) if i >= 2 and _is_magnification(t)), None)
    if view_idx is None:
        raise ValueError(
            f"{path.name!r}: no view token found (expected one of {sorted(VIEW_TOKENS)})"
        )
    if mag_idx is None:
        raise ValueError(f"{path.name!r}: no magnification token found (expected e.g. 20X)")

    head_end = min(view_idx, mag_idx)
    tail_start = max(view_idx, mag_idx) + 1
    if tail_start - head_end != 2:
        raise ValueError(
            f"{path.name!r}: view ({tokens[view_idx]!r}) and magnification "
            f"({tokens[mag_idx]!r}) must be adjacent fields"
        )

    head = tokens[2:head_end]
    if not head:
        raise ValueError(f"{path.name!r}: no genotype field between embryo id and view")

    inferred_timepoint = False
    if len(head) >= 2 and _is_timepoint(head[-1]):
        genotype = "_".join(head[:-1])
        timepoint = head[-1]
    elif len(head) == 1:
        genotype = head[0]
        if default_timepoint is None:
            raise ValueError(
                f"{path.name!r}: no timepoint field (expected e.g. 12s between "
                f"genotype and view) and no default_timepoint supplied"
            )
        timepoint = default_timepoint
        inferred_timepoint = True
    else:
        # Several tokens but the last is not a recognised timepoint: treat the
        # last as the timepoint anyway rather than silently folding it into the
        # genotype, since the spec puts a timepoint there.
        genotype = "_".join(head[:-1])
        timepoint = head[-1]

    if strict and inferred_timepoint:
        raise ValueError(f"{path.name!r}: timepoint field is missing (strict mode)")

    channels = tuple(tokens[tail_start:])
    if not channels:
        raise ValueError(f"{path.name!r}: no gene channel fields after magnification")

    return EmbryoName(
        date=date,
        embryo_num=embryo_num,
        genotype=genotype,
        timepoint=timepoint,
        view=tokens[view_idx].lower(),
        magnification=tokens[mag_idx].upper(),
        channels=channels,
        path=path,
    )


def timepoint_from_directory(directory: str | Path) -> Optional[str]:
    """Best-effort timepoint from a directory name, e.g. ``wt_lpm_nd2_12s_dorsal`` -> ``12s``."""
    for token in Path(directory).name.split("_"):
        if _is_timepoint(token):
            return token.lower()
    return None


def discover_embryos(
    input_dir: str | Path,
    default_timepoint: Optional[str] = None,
    strict: bool = False,
    pattern: str = "*.nd2",
    on_error: str = "warn",
) -> List[EmbryoName]:
    """Parse every ND2 in ``input_dir``.

    Args:
        default_timepoint: used for files missing the timepoint field.  When
            ``None``, it is inferred from the directory name.
        on_error: ``"warn"`` skips unparseable files with a message, ``"raise"``
            propagates, ``"skip"`` is silent.
    """
    input_dir = Path(input_dir)
    if default_timepoint is None:
        default_timepoint = timepoint_from_directory(input_dir)

    embryos: List[EmbryoName] = []
    for nd2_path in sorted(input_dir.glob(pattern)):
        try:
            embryos.append(parse_embryo_name(nd2_path, default_timepoint, strict=strict))
        except ValueError as exc:
            if on_error == "raise":
                raise
            if on_error == "warn":
                print(f"  [SKIP] {exc}")
    return embryos


def group_into_cohorts(embryos: Iterable[EmbryoName]) -> Dict[CohortKey, List[EmbryoName]]:
    """Bucket embryos by ``(genotype, timepoint, view, magnification)``.

    Insertion order is preserved inside each cohort so the first-listed embryo is
    a stable default registration reference.
    """
    cohorts: Dict[CohortKey, List[EmbryoName]] = {}
    for embryo in embryos:
        cohorts.setdefault(embryo.cohort, []).append(embryo)
    return cohorts


# ---------------------------------------------------------------------------
# Renaming existing files onto the canonical spec
# ---------------------------------------------------------------------------

def plan_renames(
    input_dir: str | Path,
    timepoint: Optional[str] = None,
    pattern: str = "*.nd2",
) -> List[Tuple[Path, Path]]:
    """Old -> new paths needed to bring ``input_dir`` onto the canonical spec.

    Files already canonical are omitted.  Nothing is written.
    """
    input_dir = Path(input_dir)
    if timepoint is None:
        timepoint = timepoint_from_directory(input_dir)

    planned: List[Tuple[Path, Path]] = []
    for nd2_path in sorted(input_dir.glob(pattern)):
        try:
            parsed = parse_embryo_name(nd2_path, default_timepoint=timepoint)
        except ValueError as exc:
            print(f"  [SKIP] {exc}")
            continue
        new_path = nd2_path.with_name(f"{format_embryo_name(parsed)}{nd2_path.suffix}")
        if new_path != nd2_path:
            planned.append((nd2_path, new_path))
    return planned


def apply_renames(
    planned: Sequence[Tuple[Path, Path]],
    manifest_path: Optional[str | Path] = None,
    dry_run: bool = True,
) -> List[Tuple[Path, Path]]:
    """Perform a rename plan, recording it so it can be undone.

    Refuses the whole batch if any destination already exists or if two sources
    map onto one destination -- a partial rename of a cohort is worse than none,
    because half the embryos would then parse into a different cohort.
    """
    planned = list(planned)
    if not planned:
        print("  nothing to rename; all filenames already canonical")
        return []

    destinations = [new for _, new in planned]
    duplicates = {d for d in destinations if destinations.count(d) > 1}
    if duplicates:
        raise ValueError(
            "rename plan collides: multiple sources map to "
            + ", ".join(str(d.name) for d in sorted(duplicates))
        )
    sources = {old for old, _ in planned}
    for new in destinations:
        if new.exists() and new not in sources:
            raise FileExistsError(f"destination already exists: {new}")

    for old, new in planned:
        print(f"  {old.name}\n    -> {new.name}")

    if dry_run:
        print(f"\n  DRY RUN: {len(planned)} file(s) would be renamed; nothing written")
        return planned

    if manifest_path is None:
        manifest_path = planned[0][0].parent / MANIFEST_NAME
    manifest_path = Path(manifest_path)

    # Write the manifest BEFORE touching anything, so an interrupted rename is
    # still recoverable from the record on disk.
    existing = []
    if manifest_path.exists():
        with open(manifest_path, newline="", encoding="utf-8") as handle:
            existing = [row for row in csv.DictReader(handle)]
    with open(manifest_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["old_name", "new_name"])
        writer.writeheader()
        for row in existing:
            writer.writerow(row)
        for old, new in planned:
            writer.writerow({"old_name": old.name, "new_name": new.name})

    for old, new in planned:
        old.rename(new)
    print(f"\n  renamed {len(planned)} file(s); manifest -> {manifest_path}")
    return planned


def undo_renames(manifest_path: str | Path, dry_run: bool = True) -> List[Tuple[Path, Path]]:
    """Reverse the renames recorded in a manifest, most recent first."""
    manifest_path = Path(manifest_path)
    directory = manifest_path.parent
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    reverted: List[Tuple[Path, Path]] = []
    for row in reversed(rows):
        new = directory / row["new_name"]
        old = directory / row["old_name"]
        if not new.exists():
            print(f"  [SKIP] {new.name} not present")
            continue
        print(f"  {new.name}\n    -> {old.name}")
        reverted.append((new, old))
    if dry_run:
        print(f"\n  DRY RUN: {len(reverted)} file(s) would be reverted")
        return reverted
    for new, old in reverted:
        new.rename(old)
    print(f"\n  reverted {len(reverted)} file(s)")
    return reverted
