"""The data contract between an environment and the model.

A trajectory source produces `Trajectory` objects and describes itself with an
`ObservationSpec`. Training reads `observations` and `actions` only. The
remaining fields are still recorded, and recovering them later means
regenerating the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax

class ObservationMode(Enum):
    """Which view of the environment an observation records."""

    TOP_DOWN = "top_down"
    """Full grid. The fully observed case."""

    EGOCENTRIC = "egocentric"
    """Agent-centred partial view, fixed size and oriented to the agent's
    facing, with anything behind the agent unobserved. The partially observed
    case."""


@dataclass(frozen=True)
class FieldSpec:
    """One component of an observation, with its shape and value domain.

    Exactly one of `cardinality` and `value_range` is set.

    Attributes:
        name: Identifier for this field, such as "grid" or "message".
        shape: Spatial shape of the field, excluding the channel axis.
        cardinality: Number of classes per channel for a discrete field.
        value_range: Inclusive (low, high) bounds of the stored values for a
            continuous field, before any normalisation.
    """

    name: str
    shape: tuple[int, ...]
    cardinality: tuple[int, ...] | None = None
    value_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Reject a field unless exactly one domain is declared.

        Raises:
            ValueError: If the number of domains set is not exactly one.
        """
        declared = [
            name
            for name, value in (
                ("cardinality", self.cardinality),
                ("value_range", self.value_range),
            )
            if value is not None
        ]
        if len(declared) != 1:
            raise ValueError(
                f"field {self.name!r} declares {declared or 'neither'} of "
                "cardinality and value_range. Exactly one is required: "
                "cardinality for a discrete field, value_range for a "
                "continuous one."
            )


@dataclass(frozen=True)
class ObservationSpec:
    """Everything a tokeniser needs to know about an environment.

    Attributes:
        fields: One FieldSpec per observation component, in a fixed order.
        num_actions: Size of the discrete action space, which sizes the action
            embedding table.
    """

    fields: tuple[FieldSpec, ...]
    num_actions: int

    def single_field(self) -> FieldSpec:
        """Return the only field, for environments that have exactly one.

        Returns:
            The sole FieldSpec.

        Raises:
            ValueError: If the spec carries anything other than one field.
        """
        if len(self.fields) != 1:
            raise ValueError(
                f"expected a single-field observation, got {len(self.fields)} "
                f"fields: {[f.name for f in self.fields]}. A multi-field "
                "environment needs its own Tokeniser implementation."
            )
        return self.fields[0]


@dataclass(frozen=True)
class Trajectory:  # pylint: disable=too-many-instance-attributes
    """One complete episode, recorded in every stored observation mode.

    Attributes:
        observations: Per observation mode, an array of shape
            (num_steps + 1, *field_shape, num_channels). The extra frame is the
            endpoint state reached after the final action.
        actions: Discrete action indices, shape (num_steps,).
        rewards: Shape (num_steps,).
        terminated: Shape (num_steps,). True only at a genuine absorbing state.
        truncated: Shape (num_steps,). True at a step-limit cutoff, and never
            merged with `terminated`.
        goal_position: Goal coordinates, shape (num_goals, 2) as (row, column)
            pairs.
        provenance: Seed, environment name and generating policy.
        executed_actions: What the environment actually executed, shape
            (num_steps,). Equal to `actions` unless slip was injected. Training
            reads `actions`, the commanded one.
        state: Environment state per observation index, as a flat mapping of
            leaf name to array, or None when the episode carries none. An
            episode holding state can render any view on demand; one without it
            is limited to the modes in `observations`.
    """

    observations: dict[ObservationMode, jax.Array]
    actions: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    goal_position: jax.Array
    provenance: dict[str, str | int]
    executed_actions: jax.Array
    state: dict[str, jax.Array] | None = None

    def __len__(self) -> int:
        """Return the number of actions, which is the episode length."""
        return int(self.actions.shape[0])
