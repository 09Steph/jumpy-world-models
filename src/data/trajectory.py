"""The data contract between an environment and the model.

A trajectory source produces `Trajectory` objects and describes itself with an
`ObservationSpec`.

Training reads `observations` and `actions` only. The remaining `Trajectory`
fields are recorded; recovering them later means regenerating the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax

class ObservationMode(Enum):
    """Which view of the environment an observation records.

    Both modes are generated for the same trajectory.
    """

    TOP_DOWN = "top_down"
    """Full symbolic grid. The fully observed case."""

    EGOCENTRIC = "egocentric"
    """Agent-centred partial view, fixed size and oriented to the agent's
    facing, with anything behind the agent unobserved. The partially observed
    case."""


@dataclass(frozen=True)
class FieldSpec:
    """One component of an observation, with its shape and cardinality.

    Attributes:
        name: Identifier for this field, such as "grid" or "message".
        shape: Spatial shape of the field, excluding the channel axis.
        cardinality: Number of classes per channel for a discrete field. A
            continuous field carries None and is out of scope.
    """

    name: str
    shape: tuple[int, ...]
    cardinality: tuple[int, ...] | None


@dataclass(frozen=True)
class ObservationSpec:
    """Everything a tokeniser needs to know about an environment.

    A list of fields, so a multi-field environment is representable.

    Attributes:
        fields: One FieldSpec per observation component, in a fixed order.
        num_actions: Size of the discrete action space, which sizes the action
            embedding table.
    """

    fields: tuple[FieldSpec, ...]
    num_actions: int

    def single_field(self) -> FieldSpec:
        """Return the only field, for environments that have exactly one.

        Raises on a multi-field spec instead of reading `fields[0]`.

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
class Trajectory:
    """One complete episode, recorded in both observation modes.

    Attributes:
        observations: Per observation mode, an array of shape
            (num_steps + 1, *field_shape, num_channels). The extra frame is the
            endpoint state reached after the final action.
        actions: Discrete action indices, shape (num_steps,).
        rewards: Shape (num_steps,). Unread during training, recorded so a
            later policy stage needs no regeneration.
        terminated: Shape (num_steps,). True only at a genuine absorbing state.
        truncated: Shape (num_steps,). True at a step-limit cutoff, and never
            merged with `terminated`.
        goal_position: Goal coordinates, shape (num_goals, 2) as (row, column)
            pairs. The goal axis is kept, so an environment with several
            goals keeps the distinction.
        provenance: Seed, environment name and generating policy.
    """

    observations: dict[ObservationMode, jax.Array]
    actions: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    goal_position: jax.Array
    provenance: dict[str, str | int]

    def __len__(self) -> int:
        """Return the number of actions, which is the episode length.

        Returns:
            Number of transitions in this episode.
        """
        return int(self.actions.shape[0])
