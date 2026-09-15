"""The thesis figure entry point: the registry, drawing, the manifest and publication.

`run_figures_cli` draws every registered figure its artefacts support into the
figures directory, logs each as drawn, partial, skipped or failed, and writes a
manifest of the outcomes. `run_publish_cli` copies the drawn and partial figures
of the reported split, as that manifest records them, into a destination
directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from config import (
    ATARI_LONG_HORIZON,
    FIGURES_DIR_NAME,
    OUTPUTS_DIR,
    TEST_RUN_SUFFIX,
    THESIS_HALF_WIDTH_IN,
    THESIS_TEXT_WIDTH_IN,
)
from src.eval import figure_panels as panels
from src.eval.figure_style import SPLIT_DISPLAY, VECTOR_EXTENSION, FigureStyleError
from src.eval.intermediate_figures import build_intermediate_compounding
from src.eval.plots import draw_figure
from src.utils.logging_setup import get_logger
from src.utils.paths import ensure_dir, safe_rel

logger = get_logger(__name__)

LABEL_PREFIX: str = "fig:"
DEVELOPMENT_DIR_NAME: str = "development"
FIGURES_MANIFEST_FILENAME: str = "figures_manifest.json"
TABLE_EXTENSION: str = ".csv"
DIGEST_CHUNK_BYTES: int = 1 << 20

# Outcome of one figure.
DRAWN: str = "drawn"
PARTIAL: str = "partial"
SKIPPED: str = "skipped"
FAILED: str = "failed"
PUBLISHABLE: frozenset[str] = frozenset({DRAWN, PARTIAL})


@dataclass(frozen=True)
class FigureSpec:  # pylint: disable=too-many-instance-attributes
    """One declared thesis figure."""

    figure_id: str
    label: str
    title: str
    tier: str
    build: panels.Builder
    width_in: float = THESIS_TEXT_WIDTH_IN
    development_twin: bool = False

    @property
    def filename(self) -> str:
        """The output filename: the label without its prefix, with the vector extension."""
        return self.label.removeprefix(LABEL_PREFIX) + VECTOR_EXTENSION


FIGURE_REGISTRY: tuple[FigureSpec, ...] = (
    FigureSpec("F0.1", "fig:e2_error_against_horizon", "Prediction error against horizon",
               "0", panels.build_error_against_horizon, development_twin=True),
    FigureSpec("F0.2", "fig:compounding_error", "Compounding error by condition",
               "0", panels.build_compounding_error, development_twin=True),
    FigureSpec("F0.3", "fig:e2_p2b", "Fitted exponent by arm",
               "0", panels.build_exponent_forest, development_twin=True),
    FigureSpec("F0.4", "fig:e3_slip", "Prediction error under increasing stochasticity",
               "0", panels.build_slip_ladder, development_twin=True),
    FigureSpec("F0.5", "fig:intermediate_compounding", "Intermediate-state divergence per step",
               "0", build_intermediate_compounding),
    FigureSpec("F0.6", "fig:rgb_error_against_horizon",
               "Prediction error against horizon, pixel observations",
               "0", panels.build_pixel_error, development_twin=True),
    FigureSpec("F0.7", "fig:rgb_mover_skill",
               "Mover-restricted skill against horizon, pixel observations",
               "0", panels.build_pixel_mover_skill, development_twin=True),
    FigureSpec("F0.8", "fig:long_horizon_error",
               f"Prediction error to horizon {ATARI_LONG_HORIZON}",
               "0", panels.build_long_horizon, development_twin=True),
    FigureSpec("F0.9", "fig:data_saturation", "Does the data saturate, or does the model",
               "0", panels.build_data_saturation),
    FigureSpec("F1.1", "fig:room_policy_exponents", "Fitted exponent by room and collection policy",
               "1", panels.build_room_policy_exponents),
    FigureSpec("F1.2", "fig:e1_learnability_bars", "Model against copy baseline at short horizon",
               "1", panels.build_learnability_bars),
    FigureSpec("F1.3", "fig:e4_mode_gap", "Skill score by observation mode",
               "1", panels.build_mode_gap),
    FigureSpec("F2.1", "fig:supp_layouts", "Displacement by layout", "2", panels.build_layouts),
    FigureSpec("F2.2", "fig:supp_residuals", "Fit residuals by estimator",
               "2", panels.build_residuals),
    FigureSpec("F2.3", "fig:supp_estimator_panel", "Estimator identified by arm",
               "2", panels.build_estimator_panel),
    FigureSpec("F2.4", "fig:supp_test_vs_validation", "Test against validation split",
               "2", panels.build_test_vs_validation),
    FigureSpec("F2.6", "fig:supp_per_horizon_bars", "Model error at selected horizons",
               "2", panels.build_per_horizon_bars),
    FigureSpec("F2.7", "fig:supp_atari_displacement", "Atari game selection by displacement",
               "2", panels.build_atari_displacement),
    FigureSpec("F2.8", "fig:compounding_error_alt",
               "Compounding error by condition, alternate readings",
               "2", panels.build_compounding_alt),
    FigureSpec("F2.10", "fig:supp_best_vs_final_tree", "Selected against final parameter tree",
               "2", panels.build_best_vs_final),
    FigureSpec("F3.1", "fig:atari_error_against_horizon", "Prediction error against horizon, Atari",
               "3", panels.build_atari_error, width_in=THESIS_HALF_WIDTH_IN),
    # F3.2 and A2 are not registered: no artefact carries a per-game slice of
    # the Atari evaluation.
    FigureSpec("F3.5", "fig:atari_horizon_ablation",
               "Effect of training horizon on long-horizon error",
               "3", panels.build_horizon_ablation),
    FigureSpec("F3.6", "fig:atari_checkpoint_ladder",
               "Prediction error against behaviour-policy quality",
               "3", panels.build_checkpoint_ladder),
    FigureSpec("A1", "fig:skill_across_representations", "Skill score across representations",
               "A", panels.build_skill_across_representations),
)


@dataclass(frozen=True)
class FiguresRequest:
    """What one --figures invocation draws, and from which tree."""

    outputs_root: Path = OUTPUTS_DIR
    atari_results: Path = panels.ATARI_RESULTS_DIR
    run_filter: str | None = None
    dry_run: bool = False
    development: bool = False
    reps: int | None = None


@dataclass(frozen=True)
class FigureOutcome:  # pylint: disable=too-many-instance-attributes
    """What happened to one figure, as the manifest records it."""

    figure_id: str
    label: str
    filename: str
    status: str
    split: str
    reason: str = ""
    sources: tuple[str, ...] = ()
    keys: tuple[str, ...] = ()
    omissions: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    provenance: str = ""
    table: str = ""


class FiguresFailedError(panels.FigureContractError):
    """One or more figures failed; every other figure was still drawn."""

    def __init__(self, outcomes: tuple[FigureOutcome, ...]) -> None:
        """Record every outcome, naming the failed figures in the message."""
        failed = [outcome.label for outcome in outcomes if outcome.status == FAILED]
        super().__init__(f"{len(failed)} figure(s) failed: {', '.join(failed)}")
        self.outcomes = outcomes


def naming_table() -> str:
    """Return the figure ID, label, filename, width and tier of every figure as a Markdown table."""
    lines = [
        "| ID | LaTeX label | Output file | Width (in) | Tier |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| {spec.figure_id} | `{spec.label}` | `{spec.filename}` | "
        f"{spec.width_in:g} | {spec.tier} |"
        for spec in FIGURE_REGISTRY
    )
    return "\n".join(lines)


def runs_of(sources: tuple[str, ...]) -> tuple[str, ...]:
    """Return the run names cited artefacts sit under, in first-read order."""
    runs = [
        parts[1] for parts in (source.split("/") for source in sources)
        if parts[0] == OUTPUTS_DIR.name and len(parts) > 2
    ]
    return tuple(dict.fromkeys(runs))


def _provenance(
    spec: FigureSpec, split: str, build: panels.FigureBuild, sources: tuple[str, ...]
) -> str:
    """Return the caption provenance line: the figure, its split, sources, keys and notes."""
    notes = "".join(f" Note: {note}." for note in build.notes)
    return (
        f"Figure {spec.figure_id}. {spec.label}. {SPLIT_DISPLAY.get(split, split)}. "
        f"Sources: {'; '.join(sources)}. Keys: {', '.join(build.keys)}. "
        f"{panels.INTERVAL_DESCRIPTION}.{notes}"
    )


def _write_table(rows: tuple[dict, ...], path: Path) -> Path:
    """Write a figure's plotted values beside it, one row per plotted point."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("wrote figure table -> %s", safe_rel(path))
    return path


def _render(  # pylint: disable=too-many-arguments
    spec: FigureSpec,
    tree: panels.OutputsTree,
    split: str,
    path: Path,
    request: FiguresRequest,
) -> FigureOutcome | None:
    """Build and draw one figure, returning its outcome, or None when filtered out.

    Skipped and failed figures are never filtered out.
    """
    tree.reads.clear()
    base = {
        "figure_id": spec.figure_id, "label": spec.label, "filename": spec.filename,
        "split": split,
    }
    try:
        build = spec.build(tree, split)
    except panels.MissingArtefactError as error:
        logger.warning("figure %s %s SKIPPED -- %s", spec.figure_id, spec.label, error)
        return FigureOutcome(**base, status=SKIPPED, reason=str(error), sources=tuple(tree.reads))
    except (panels.FigureContractError, FigureStyleError) as error:
        logger.error("figure %s %s FAILED -- %s", spec.figure_id, spec.label, error)
        return FigureOutcome(**base, status=FAILED, reason=str(error), sources=tuple(tree.reads))
    sources = tuple(tree.reads)
    runs = runs_of(sources)
    if request.run_filter is not None and not {
        request.run_filter, request.run_filter + TEST_RUN_SUFFIX
    } & set(runs):
        return None
    for omission in build.omissions:
        logger.warning("figure %s %s omits: %s", spec.figure_id, spec.label, omission)
    provenance = _provenance(spec, split, build, sources)
    status = PARTIAL if build.omissions else DRAWN
    outcome = FigureOutcome(
        **base, status=status, sources=sources, keys=build.keys, omissions=build.omissions,
        notes=build.notes, provenance=provenance,
    )
    if request.dry_run:
        logger.info(
            "figure %s %s would draw split=%s runs=%s -> %s",
            spec.figure_id, spec.label, split, ",".join(runs), safe_rel(path),
        )
        return outcome
    layout = replace(build.layout, title=spec.title, width_in=spec.width_in, subject=provenance)
    try:
        draw_figure(layout, ensure_dir(path.parent) / path.name)
    except FigureStyleError as error:
        logger.error("figure %s %s FAILED -- %s", spec.figure_id, spec.label, error)
        return replace(outcome, status=FAILED, reason=str(error))
    if build.table:
        outcome = replace(
            outcome, table=_write_table(build.table, path.with_suffix(TABLE_EXTENSION)).name
        )
    logger.info(
        "figure %s %s %s split=%s runs=%s -> %s",
        spec.figure_id, spec.label, status, split, ",".join(runs), safe_rel(path),
    )
    return outcome


def _draw_development(tree: panels.OutputsTree, target: Path, dry_run: bool) -> int:
    """Draw every discovered run's per-metric figures, returning the count drawn.

    A dry run counts without drawing, and an error stops the rest of that run's
    figures.
    """
    count = 0
    for run in panels.discover_runs(tree.root):
        try:
            layouts = panels.development_layouts(tree, run)
            for stem, layout in layouts:
                path = target / run.run / run.env / f"{stem}{VECTOR_EXTENSION}"
                if not dry_run:
                    draw_figure(layout, ensure_dir(path.parent) / path.name)
                count += 1
        except (panels.MissingArtefactError, panels.FigureContractError, FigureStyleError) as error:
            logger.warning("development figures for %s/%s stopped -- %s", run.run, run.env, error)
    return count


def _report_coverage(tree: panels.OutputsTree, outcomes: list[FigureOutcome]) -> None:
    """Log every discovered test run no reported figure read."""
    read = {run for outcome in outcomes for run in runs_of(outcome.sources)}
    for run in panels.discover_runs(tree.root):
        if run.run.endswith(TEST_RUN_SUFFIX) and run.run not in read:
            logger.info("run %s/%s is read by no reported figure", run.run, run.env)


def _write_manifest(target: Path, outcomes: list[FigureOutcome]) -> Path:
    """Write the manifest of every figure outcome."""
    path = ensure_dir(target) / FIGURES_MANIFEST_FILENAME
    payload = {"figures": [asdict(outcome) for outcome in outcomes]}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote figure manifest -> %s", safe_rel(path))
    return path


def run_figures_cli(request: FiguresRequest) -> tuple[FigureOutcome, ...]:
    """Draw every registered figure the tree supports and write the manifest.

    With a run filter the manifest holds only the figures that read that run,
    plus every skipped or failed figure, so a later publish copies only those. A
    dry run draws nothing and writes no manifest.

    Args:
        request: The tree to read, and what to draw from it.

    Returns:
        One outcome per figure not filtered out, development twins included.

    Raises:
        FiguresFailedError: If any figure failed, after every other figure is drawn.
    """
    tree = panels.OutputsTree(
        request.outputs_root, request.atari_results,
        reps=panels.BOOTSTRAP_REPS if request.reps is None else request.reps,
    )
    target = request.outputs_root / FIGURES_DIR_NAME
    outcomes: list[FigureOutcome] = []
    for spec in FIGURE_REGISTRY:
        outcome = _render(spec, tree, panels.REPORTED_SPLIT, target / spec.filename, request)
        if outcome is not None:
            outcomes.append(outcome)
    if request.development:
        development = target / DEVELOPMENT_DIR_NAME
        for spec in FIGURE_REGISTRY:
            if spec.development_twin:
                twin = _render(
                    spec, tree, panels.VALIDATION_SPLIT, development / spec.filename, request
                )
                if twin is not None:
                    outcomes.append(twin)
        drawn = _draw_development(tree, development, request.dry_run)
        logger.info("development figures: %d", drawn)
    _report_coverage(tree, outcomes)
    counts = Counter(outcome.status for outcome in outcomes)
    logger.info(
        "figures: %d drawn, %d partial, %d skipped, %d failed, of %d",
        counts[DRAWN], counts[PARTIAL], counts[SKIPPED], counts[FAILED], len(outcomes),
    )
    if not request.dry_run:
        _write_manifest(target, outcomes)
    if counts[FAILED]:
        raise FiguresFailedError(tuple(outcomes))
    return tuple(outcomes)


def _digest(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DIGEST_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_publish_cli(destination: Path, outputs_root: Path = OUTPUTS_DIR) -> Counter:
    """Copy the drawn and partial reported figures in the manifest into a directory.

    A figure the manifest does not mark publishable leaves any earlier copy in
    the destination untouched. Figure tables are not copied.

    Args:
        destination: An existing directory outside the outputs tree.
        outputs_root: The tree the figures were drawn into.

    Returns:
        How many figures were new, updated or unchanged.

    Raises:
        FileNotFoundError: If the manifest or a figure it names is absent.
        NotADirectoryError: If the destination is not a directory.
        ValueError: If the destination sits inside the outputs tree.
    """
    source_dir = outputs_root / FIGURES_DIR_NAME
    manifest = source_dir / FIGURES_MANIFEST_FILENAME
    if not manifest.is_file():
        raise FileNotFoundError(f"{safe_rel(manifest)} is absent; run --figures first")
    if not destination.is_dir():
        raise NotADirectoryError(f"--publish destination {destination.name} is not a directory")
    if destination.resolve().is_relative_to(outputs_root.resolve()):
        raise ValueError(
            "--publish copies out of outputs/, so its destination cannot sit inside it"
        )
    declared = {spec.label: spec for spec in FIGURE_REGISTRY}
    entries = json.loads(manifest.read_text(encoding="utf-8"))["figures"]
    counts: Counter = Counter()
    for entry in entries:
        spec = declared.get(entry["label"])
        reported = entry["split"] == panels.REPORTED_SPLIT
        if spec is None or entry["status"] not in PUBLISHABLE or not reported:
            continue
        source = source_dir / spec.filename
        if not source.is_file():
            raise FileNotFoundError(f"{safe_rel(source)} is named in the manifest and absent")
        target = destination / spec.filename
        state = "new"
        if target.is_file():
            state = "unchanged" if _digest(target) == _digest(source) else "updated"
        if state != "unchanged":
            target.write_bytes(source.read_bytes())
        counts[state] += 1
        logger.info("publish %s %s %s", spec.figure_id, spec.label, state)
    logger.info(
        "published into %s: %d new, %d updated, %d unchanged",
        destination.name, counts["new"], counts["updated"], counts["unchanged"],
    )
    return counts
