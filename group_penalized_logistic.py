from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _safe_logit(p: float) -> float:
    clipped = float(np.clip(p, 1e-6, 1.0 - 1e-6))
    return float(np.log(clipped / (1.0 - clipped)))


def _group_index_map(group_ids: np.ndarray) -> list[tuple[int, np.ndarray]]:
    unique_groups = np.unique(group_ids)
    return [(int(group), np.flatnonzero(group_ids == group)) for group in unique_groups]


def _group_weights(group_map: list[tuple[int, np.ndarray]], mode: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for group, idx in group_map:
        size = max(int(len(idx)), 1)
        if mode == "sqrt":
            weights[group] = float(np.sqrt(size))
        elif mode == "none":
            weights[group] = 1.0
        elif mode == "inverse_size":
            weights[group] = 1.0 / float(size)
        elif mode == "inverse_sqrt":
            weights[group] = 1.0 / float(np.sqrt(size))
        else:
            raise ValueError(f"Unsupported group weight mode: {mode}")
    return weights


def _logistic_loss_and_grad(
    X: np.ndarray,
    y: np.ndarray,
    coef: np.ndarray,
    intercept: float,
) -> tuple[float, np.ndarray, float]:
    scores = X @ coef + intercept
    prob = expit(scores)
    loss = float(np.mean(np.logaddexp(0.0, scores) - y * scores))
    grad_coef = (X.T @ (prob - y)) / X.shape[0]
    grad_intercept = float(np.mean(prob - y))
    return loss, grad_coef, grad_intercept


def _group_lasso_penalty(
    coef: np.ndarray,
    group_map: list[tuple[int, np.ndarray]],
    weights: dict[int, float],
) -> float:
    return float(sum(weights[group] * np.linalg.norm(coef[idx]) for group, idx in group_map))


def _group_ridge_penalty(
    coef: np.ndarray,
    group_map: list[tuple[int, np.ndarray]],
    weights: dict[int, float],
) -> float:
    return float(sum(weights[group] * float(coef[idx] @ coef[idx]) for group, idx in group_map))


def _prox_group_lasso(
    coef: np.ndarray,
    step_size: float,
    reg_strength: float,
    group_map: list[tuple[int, np.ndarray]],
    weights: dict[int, float],
) -> np.ndarray:
    updated = coef.copy()
    for group, idx in group_map:
        block = updated[idx]
        block_norm = float(np.linalg.norm(block))
        threshold = step_size * reg_strength * weights[group]
        if block_norm <= threshold:
            updated[idx] = 0.0
            continue
        updated[idx] = (1.0 - threshold / block_norm) * block
    return updated


def _logistic_lipschitz_constant(X: np.ndarray) -> float:
    n_samples = max(int(X.shape[0]), 1)
    x_aug = np.column_stack([X, np.ones(n_samples, dtype=float)])
    spectral_norm = float(np.linalg.norm(x_aug, ord=2))
    return max(0.25 * spectral_norm * spectral_norm / n_samples, 1e-6)


def _group_norms(coef: np.ndarray, group_map: list[tuple[int, np.ndarray]]) -> dict[int, float]:
    return {group: float(np.linalg.norm(coef[idx])) for group, idx in group_map}


def _active_mask_from_groups(
    group_map: list[tuple[int, np.ndarray]],
    active_groups: list[int],
    n_features: int,
) -> np.ndarray:
    mask = np.zeros(n_features, dtype=bool)
    active = set(active_groups)
    for group, idx in group_map:
        if group in active:
            mask[idx] = True
    return mask


@dataclass
class PenalizedLogisticResult:
    coef_: np.ndarray
    intercept_: float
    objective_: float
    n_iter_: int
    converged_: bool
    group_norms_: dict[int, float]
    active_groups_: list[int]
    active_mask_: np.ndarray

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return expit(X @ self.coef_ + self.intercept_)


def fit_group_lasso_logistic(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    group_reg: float,
    *,
    group_weight_mode: str = "sqrt",
    max_iter: int = 4000,
    tol: float = 1e-6,
    selection_tol: float = 1e-8,
) -> PenalizedLogisticResult:
    group_map = _group_index_map(group_ids)
    weights = _group_weights(group_map, group_weight_mode)
    lipschitz = _logistic_lipschitz_constant(X)
    step = 1.0 / lipschitz

    coef = np.zeros(X.shape[1], dtype=float)
    intercept = _safe_logit(float(np.mean(y)))
    accel_coef = coef.copy()
    accel_intercept = intercept
    accel_t = 1.0

    prev_objective = float("inf")
    converged = False
    objective = prev_objective

    for iter_idx in range(1, max_iter + 1):
        loss, grad_coef, grad_intercept = _logistic_loss_and_grad(
            X,
            y,
            accel_coef,
            accel_intercept,
        )
        next_coef = _prox_group_lasso(
            accel_coef - step * grad_coef,
            step,
            group_reg,
            group_map,
            weights,
        )
        next_intercept = accel_intercept - step * grad_intercept

        objective = (
            _logistic_loss_and_grad(X, y, next_coef, next_intercept)[0]
            + group_reg * _group_lasso_penalty(next_coef, group_map, weights)
        )

        max_delta = max(
            float(np.max(np.abs(next_coef - coef))) if next_coef.size else 0.0,
            abs(next_intercept - intercept),
        )
        rel_obj = abs(prev_objective - objective) / max(abs(prev_objective), 1.0)
        if max_delta < tol or rel_obj < tol:
            coef = next_coef
            intercept = next_intercept
            converged = True
            break

        next_t = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * accel_t * accel_t))
        momentum = (accel_t - 1.0) / next_t
        accel_coef = next_coef + momentum * (next_coef - coef)
        accel_intercept = next_intercept + momentum * (next_intercept - intercept)
        coef = next_coef
        intercept = next_intercept
        accel_t = next_t
        prev_objective = objective
    else:
        iter_idx = max_iter

    norms = _group_norms(coef, group_map)
    active_groups = [group for group, norm in norms.items() if norm > selection_tol]
    active_mask = _active_mask_from_groups(group_map, active_groups, len(coef))
    return PenalizedLogisticResult(
        coef_=coef,
        intercept_=float(intercept),
        objective_=float(objective),
        n_iter_=int(iter_idx),
        converged_=bool(converged),
        group_norms_=norms,
        active_groups_=active_groups,
        active_mask_=active_mask,
    )


def fit_group_ridge_logistic(
    X: np.ndarray,
    y: np.ndarray,
    group_ids: np.ndarray,
    ridge_reg: float,
    *,
    group_weight_mode: str = "none",
    max_iter: int = 2000,
    tol: float = 1e-8,
) -> PenalizedLogisticResult:
    group_map = _group_index_map(group_ids)
    weights = _group_weights(group_map, group_weight_mode)

    def objective_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        coef = params[:-1]
        intercept = float(params[-1])
        loss, grad_coef, grad_intercept = _logistic_loss_and_grad(X, y, coef, intercept)
        penalty = 0.5 * ridge_reg * _group_ridge_penalty(coef, group_map, weights)
        penalty_grad = np.zeros_like(coef)
        for group, idx in group_map:
            penalty_grad[idx] += ridge_reg * weights[group] * coef[idx]
        grad = np.concatenate([grad_coef + penalty_grad, np.array([grad_intercept])])
        return float(loss + penalty), grad

    init_params = np.zeros(X.shape[1] + 1, dtype=float)
    init_params[-1] = _safe_logit(float(np.mean(y)))

    result = minimize(
        fun=lambda params: objective_and_grad(params)[0],
        x0=init_params,
        jac=lambda params: objective_and_grad(params)[1],
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": float(tol)},
    )

    coef = result.x[:-1]
    intercept = float(result.x[-1])
    norms = _group_norms(coef, group_map)
    active_groups = [group for group, norm in norms.items() if norm > 1e-10]
    active_mask = _active_mask_from_groups(group_map, active_groups, len(coef))
    return PenalizedLogisticResult(
        coef_=coef,
        intercept_=intercept,
        objective_=float(result.fun),
        n_iter_=int(result.nit),
        converged_=bool(result.success),
        group_norms_=norms,
        active_groups_=active_groups,
        active_mask_=active_mask,
    )
