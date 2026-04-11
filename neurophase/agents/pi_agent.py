"""π-calculus trading agent.

Process-calculus-inspired agent framework. Each agent holds a PiRule —
a set of numerical conditions and categorical actions — and exposes four
primitives:

    mutation(x)   — perturb rule conditions to explore a neighbourhood
    repair(y)     — revert to the last stable rule
    clone()       — deep-copy the rule (propagation)
    learn(z)      — update rule with market context

The behavioural cycle is:

    1. Read the current market state and emergent-phase reading.
    2. If instability is detected, mutate and A/B test the candidate.
    3. Retain the candidate if its efficiency E(G) exceeds the baseline:
           E(G) = Sharpe(G) + λ · Stability(G)
    4. Otherwise, repair to the last good rule.
    5. Store (context, rule, efficiency) in semantic memory.

Everything is in-memory, deterministic, and dependency-free — this is
the cognitive skeleton, not a full trading engine.

Ported from the π-system reference (section 4, "π-Agent Behavior")
with strict typing and honest-null failure modes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Final

import numpy as np

_DEFAULT_STABILITY_WEIGHT: Final[float] = 0.3
_DEFAULT_MUTATION_STD: Final[float] = 0.1


@dataclass
class PiRule:
    """A mutable rule — numerical conditions plus a string action.

    Conditions hold hyper-parameters controlling the agent's behaviour
    (thresholds, weights, decay constants). The action is a categorical
    tag like ``"enter_long"`` or ``"hold"``.
    """

    conditions: dict[str, float]
    action: str

    def copy(self) -> PiRule:
        """Deep-copy of the rule (independent mutable state)."""
        return PiRule(conditions=dict(self.conditions), action=self.action)

    def generate_mutation(
        self,
        rng: np.random.Generator,
        std: float = _DEFAULT_MUTATION_STD,
    ) -> PiRule:
        """Return a mutated copy: Gaussian perturbation on numerical conditions."""
        if std < 0:
            raise ValueError(f"std must be non-negative, got {std}")
        mutated = dict(self.conditions)
        for key in mutated:
            mutated[key] = float(mutated[key] + rng.normal(0.0, std))
        return PiRule(conditions=mutated, action=self.action)


@dataclass(frozen=True)
class AgentEfficiency:
    """Per-trial efficiency measurement used for A/B testing.

    E(G) = Sharpe + λ · Stability
    """

    sharpe: float
    stability: float
    weight: float = _DEFAULT_STABILITY_WEIGHT

    @property
    def value(self) -> float:
        return float(self.sharpe + self.weight * self.stability)


@dataclass(frozen=True)
class MarketContext:
    """Context handed to the agent at each evaluation step.

    Attributes
    ----------
    R : float
        Current Kuramoto order parameter.
    dH : float
        Shannon entropy change ΔH_S(t).
    kappa : float
        Mean Ricci curvature κ̄(t).
    ism : float
        Information-Structural Metric value.
    regime_tag : str
        Categorical regime label (e.g. ``"trend"``, ``"chop"``, ``"vol-spike"``).
    """

    R: float
    dH: float
    kappa: float
    ism: float
    regime_tag: str = "unknown"


@dataclass
class SemanticMemory:
    """Append-only memory of (context, rule, efficiency) triples.

    The retrieve method selects triples whose context-vector is close to
    the query via cosine similarity.
    """

    store: list[tuple[MarketContext, PiRule, AgentEfficiency]] = field(default_factory=list)

    def add(self, context: MarketContext, rule: PiRule, efficiency: AgentEfficiency) -> None:
        """Store a triple."""
        self.store.append((context, rule.copy(), efficiency))

    def retrieve(
        self, query: MarketContext, threshold: float = 0.8, limit: int = 5
    ) -> list[tuple[MarketContext, PiRule, AgentEfficiency]]:
        """Return triples with cosine similarity ≥ ``threshold``.

        Similarity is computed over the numerical slice of the context
        (R, dH, kappa, ism). Regime tag is ignored for retrieval.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        q_vec = np.array([query.R, query.dH, query.kappa, query.ism], dtype=np.float64)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm == 0.0:
            return []
        ranked: list[tuple[float, tuple[MarketContext, PiRule, AgentEfficiency]]] = []
        for context, rule, eff in self.store:
            v = np.array(
                [context.R, context.dH, context.kappa, context.ism],
                dtype=np.float64,
            )
            v_norm = float(np.linalg.norm(v))
            if v_norm == 0.0:
                continue
            sim = float(np.dot(q_vec, v) / (q_norm * v_norm))
            if sim >= threshold:
                ranked.append((sim, (context, rule, eff)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [triple for _sim, triple in ranked[:limit]]


@dataclass
class PiAgent:
    """π-calculus trading agent with mutable rule + memory.

    Parameters
    ----------
    rule : PiRule
        Initial rule.
    memory : SemanticMemory
        Shared or per-agent memory.
    stability_weight : float
        λ in E(G) = Sharpe + λ · Stability.
    seed : int | None
        PRNG seed for reproducible mutation.
    """

    rule: PiRule
    memory: SemanticMemory = field(default_factory=SemanticMemory)
    stability_weight: float = _DEFAULT_STABILITY_WEIGHT
    seed: int | None = None
    _rng: np.random.Generator = field(init=False)
    _last_stable: PiRule = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._last_stable = self.rule.copy()

    # ───────────────────────── primitives ─────────────────────────

    def mutate(self, std: float = _DEFAULT_MUTATION_STD) -> PiRule:
        """Return a mutated candidate rule (does not modify self)."""
        return self.rule.generate_mutation(self._rng, std=std)

    def repair(self) -> None:
        """Revert to the last rule flagged as stable."""
        self.rule = self._last_stable.copy()

    def clone(self) -> PiAgent:
        """Deep-copy of this agent (independent memory + rule)."""
        new = PiAgent(
            rule=self.rule.copy(),
            memory=SemanticMemory(store=copy.deepcopy(self.memory.store)),
            stability_weight=self.stability_weight,
            seed=self.seed,
        )
        return new

    def learn(self, context: MarketContext, efficiency: AgentEfficiency) -> None:
        """Commit a (context, rule, efficiency) triple to memory."""
        self.memory.add(context, self.rule, efficiency)

    # ───────────────────────── evaluation ─────────────────────────

    def step(
        self,
        context: MarketContext,
        baseline: AgentEfficiency,
        candidate_efficiency: AgentEfficiency,
        candidate_rule: PiRule,
    ) -> PiRule:
        """Execute one mutation + A/B test + memory update cycle.

        The caller is responsible for running a backtest and producing
        the efficiency numbers. This method wires the decision: keep
        the candidate if it strictly beats the baseline, otherwise
        repair to the last stable rule. Either way, the chosen rule is
        committed to memory alongside the context.
        """
        if candidate_efficiency.value > baseline.value:
            self.rule = candidate_rule.copy()
            self._last_stable = candidate_rule.copy()
            self.learn(context, candidate_efficiency)
        else:
            self.repair()
            self.learn(context, baseline)
        return self.rule
