"""Reconciled per-asset attribution of an observed cash/share execution ledger.

This module consumes physical holdings and fills; it does not reconstruct a
frictionless strategy. Currency P&L and executed turnover are divided by the
session's initial equity. Exposure means weight the holdings at the start of
each marked interval by that interval's duration. Terminal liquidation is a
fill cost and has no subsequent holding interval.

The functions deliberately accept the ledger by its documented attributes.
They can inspect a preserved execution implementation without importing its
package or changing its accounting. No predictor, risk or cost model is fit.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

__all__ = ["execution_attribution", "paired_execution_attribution"]


def _positive_scalar(value, name, *, allow_zero=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
        raise ValueError(f"{name} must be a finite {'nonnegative' if allow_zero else 'positive'} scalar")
    return float(value)


def _array(value, shape, name, *, finite=True):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != shape:
        raise ValueError(f"{name} must be a numeric array of shape {shape}")
    result = np.asarray(raw, dtype=float)
    if finite and not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _durations(times):
    if times.ndim != 1 or len(times) < 2:
        raise ValueError("ledger times must contain at least two interval boundaries")
    if times.dtype.kind == "M":
        if np.isnat(times).any():
            raise ValueError("ledger times cannot contain NaT")
        delta = np.diff(times) / np.timedelta64(1, "s")
    elif times.dtype.kind in "iuf":
        if not np.isfinite(times).all():
            raise ValueError("ledger times must be finite")
        delta = np.diff(times.astype(np.longdouble))
    elif times.dtype.kind == "O":
        try:
            delta = [(right-left).total_seconds() for left, right in zip(times[:-1], times[1:])]
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("ledger times must use one comparable timestamp basis") from error
    else:
        raise ValueError("ledger times must be numeric seconds or datetimes")
    delta = np.asarray(delta, dtype=float)
    if not np.isfinite(delta).all() or np.any(delta <= 0):
        raise ValueError("ledger times must be strictly increasing")
    return delta


def _reconcile(actual, expected, name):
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f"nonfinite accounting value in {name}")
    try:
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-8,
                                   err_msg=f"attribution does not reconcile {name}")
    except AssertionError as error:
        raise ValueError(f"attribution does not reconcile {name}") from error
    return float(np.max(np.abs(np.asarray(actual)-np.asarray(expected)), initial=0.))


def _prepare(ledger, prices, beta, beta_bound, tolerance):
    tolerance = _positive_scalar(tolerance, "tolerance", allow_zero=True)
    initial = _positive_scalar(ledger.initial_equity, "initial_equity")
    times, assets = np.asarray(ledger.times), np.asarray(ledger.assets)
    if assets.ndim != 1 or not len(assets):
        raise ValueError("ledger assets must be a nonempty one-dimensional axis")
    labels = [str(asset) for asset in assets]
    if (len(set(labels)) != len(labels) or any(asset is None for asset in assets)
            or any(isinstance(asset, Real) and not np.isfinite(asset) for asset in assets)):
        raise ValueError("ledger assets must be unique nonmissing labels")
    duration = _durations(times)
    shape = (len(times), len(assets))
    if hasattr(prices, "times") and hasattr(prices, "assets") and hasattr(prices, "values"):
        if not np.array_equal(prices.times, times) or not np.array_equal(prices.assets, assets):
            raise ValueError("price-panel axes differ from the execution ledger")
        prices = prices.values
    marks = _array(prices, shape, "prices")
    if np.any(marks <= 0):
        raise ValueError("accounting prices must be positive")
    shares = _array(ledger.shares, shape, "shares")
    filled = _array(ledger.filled_shares, shape, "filled_shares")
    fill_prices = _array(ledger.fill_prices, shape, "fill_prices", finite=False)
    traded = filled != 0
    if not np.isfinite(fill_prices[traded]).all() or np.any(fill_prices[traded] <= 0):
        raise ValueError("executed shares must have finite positive fill prices")
    paid = np.where(traded, fill_prices, 0.)
    rates = _array(ledger.commission_rates, shape, "commission_rates")
    if np.any(rates < 0):
        raise ValueError("commission rates cannot be negative")
    held = _array(ledger.posttrade_weights, shape, "posttrade_weights")
    post_nav = _array(ledger.posttrade_nav, (len(times),), "posttrade_nav")
    if np.any(post_nav <= 0):
        raise ValueError("posttrade NAV must be positive")
    _reconcile(held, shares*marks/post_nav[:, None], "marked holding weights")
    nav = _array(ledger.nav, (len(times),), "nav")
    if np.any(nav <= 0):
        raise ValueError("NAV must be positive")
    pre_nav = _array(ledger.pretrade_nav, (len(times),), "pretrade_nav")
    if np.any(pre_nav <= 0):
        raise ValueError("pretrade NAV must be positive")
    if beta is not None:
        factors = _array(beta, (len(assets),), "beta")
    else:
        factors = None
    if beta_bound is not None:
        beta_bound = _positive_scalar(beta_bound, "beta_bound", allow_zero=True)
        if factors is None:
            raise ValueError("beta_bound requires supplied beta")
    decisions = np.asarray(ledger.target_decisions)
    if decisions.ndim != 1 or not len(decisions):
        raise ValueError("target_decisions must be a nonempty timestamp axis")
    targets = _array(ledger.target_weights, (len(decisions), len(assets)), "target_weights")
    try:
        decision_rows = np.searchsorted(times, decisions)
    except (TypeError, ValueError) as error:
        raise ValueError("target decision times must be on the ledger clock") from error
    if (np.any(decision_rows >= len(times)) or np.any(np.diff(decision_rows) <= 0)
            or not np.array_equal(times[decision_rows], decisions)):
        raise ValueError("target decision times must be unique increasing ledger boundaries")
    return dict(initial=initial, times=times, assets=assets, labels=labels,
                durations=duration, prices=marks, shares=shares, filled=filled,
                fill_prices=paid, rates=rates, held=held, post_nav=post_nav,
                nav=nav, pre_nav=pre_nav, beta=factors, beta_bound=beta_bound,
                tolerance=tolerance, decisions=decisions, targets=targets)


def _event_diagnostics(ledger, data, cost_by_mark, turnover_by_mark):
    """Classify fills by their sending decision, with closing orders separate."""
    times, decisions = data["times"], data["decisions"]
    n, count = len(data["assets"]), len(times)
    terminal = np.zeros(count, dtype=bool)
    boundary = np.zeros(count, dtype=bool)
    recorded_fills = np.zeros((count, n))
    ordinary_directions = [[] for _ in range(n)]
    boundary_count = np.zeros(n, dtype=int)
    seen = set()
    numerical_trade_tolerance = 1e-12
    if data["beta_bound"] is None:
        boundary_targets = np.zeros(len(decisions), dtype=bool)
    else:
        exposure = data["targets"]@data["beta"]
        boundary_targets = np.abs(np.abs(exposure)-data["beta_bound"]) <= data["tolerance"]
    for record in ledger.order_records:
        row = int(np.searchsorted(times, record.arrival_time))
        if row >= count or times[row] != record.arrival_time:
            raise ValueError("an order arrival is absent from the ledger clock")
        if row in seen:
            raise ValueError("multiple fill records at one mark need a fill-level attribution ledger")
        seen.add(row)
        quantities = _array(record.filled_shares, (n,), "order filled_shares")
        recorded_fills[row] = quantities
        terminal[row] = bool(record.terminal)
        if record.terminal:
            continue
        j = int(np.searchsorted(decisions, record.decision_time))
        if j >= len(decisions) or decisions[j] != record.decision_time:
            raise ValueError("a nonterminal order is not bound to a target decision")
        boundary[row] = boundary_targets[j]
        nonzero = np.abs(quantities*data["fill_prices"][row])/data["initial"] > numerical_trade_tolerance
        for i in np.flatnonzero(nonzero):
            ordinary_directions[i].append(int(np.sign(quantities[i])))
        if boundary[row]:
            boundary_count += nonzero
    _reconcile(recorded_fills, data["filled"], "order records and filled-share arrays")
    reversals = np.asarray([sum(left != right for left, right in zip(sequence[:-1], sequence[1:]))
                            for sequence in ordinary_directions], dtype=int)
    return dict(
        terminal_cost=cost_by_mark[terminal].sum(axis=0),
        terminal_turnover=turnover_by_mark[terminal].sum(axis=0),
        boundary_cost=cost_by_mark[boundary].sum(axis=0),
        boundary_turnover=turnover_by_mark[boundary].sum(axis=0),
        trade_sign_reversals=reversals,
        nonterminal_fill_events=np.asarray([len(s) for s in ordinary_directions]),
        boundary_fill_events=boundary_count,
        boundary_target_decisions=int(boundary_targets.sum()),
        terminal_order_events=int(terminal.sum()),
        numerical_trade_tolerance=numerical_trade_tolerance,
    )


def execution_attribution(ledger, prices, *, beta=None, beta_bound=None, tolerance=1e-8):
    """Return JSON-safe per-asset realized P&L, costs, turnover and exposures.

    ``beta`` is an already-known static vector in ledger asset order. With a
    supplied ``beta_bound``, a boundary trade means a nonterminal fill whose
    originating *target* satisfies ``abs(abs(beta @ target)-bound) <= tolerance``.
    It is a descriptive event label and does not attribute a trade causally to
    the bound. Sign reversals count changes in nonterminal filled direction,
    ignoring fills smaller than ``1e-12`` of initial equity. A reversal may be
    a normal rebalance; it does not by itself establish excessive trading.
    """
    data = _prepare(ledger, prices, beta, beta_bound, tolerance)
    initial, marks = data["initial"], data["prices"]
    fill_dollars = data["filled"]*data["fill_prices"]
    arrival = data["filled"]*np.where(data["filled"] != 0,
                                    data["fill_prices"]-marks, 0.)
    commissions = data["rates"]*np.abs(fill_dollars)
    costs = arrival+commissions
    turnover = np.abs(fill_dollars)/initial
    gross = (data["shares"][:-1]*np.diff(marks, axis=0)).sum(axis=0)
    errors = dict(
        arrival_slippage=_reconcile(arrival.sum(axis=1),
            _array(ledger.arrival_slippage, (len(marks),), "arrival_slippage"), "arrival slippage"),
        commissions=_reconcile(commissions.sum(axis=1),
            _array(ledger.commissions, (len(marks),), "commissions"), "commissions"),
        traded_dollars=_reconcile(fill_dollars,
            _array(ledger.trade_notional, marks.shape, "trade_notional"), "fill dollars"),
    )
    cost_asset = costs.sum(axis=0)
    total_net = float(gross.sum()-cost_asset.sum())
    errors["net_pnl"] = _reconcile(total_net, data["nav"][-1]-initial, "session net P&L")
    errors["reported_net_return"] = _reconcile(total_net/initial,
        float(ledger.total_return), "reported total return")
    net_path = _array(ledger.net_returns, (len(marks)-1,), "net_returns")
    errors["nav_compounding"] = _reconcile(data["nav"][1:],
        data["nav"][:-1]*(1+net_path), "NAV compounding")
    recorded_gross = _array(ledger.gross_returns, (len(marks)-1,), "gross_returns")
    interval_gross = (data["shares"][:-1]*np.diff(marks, axis=0)).sum(axis=1)/data["pre_nav"][:-1]
    errors["gross_intervals"] = _reconcile(interval_gross, recorded_gross, "gross return intervals")
    diagnostics = _event_diagnostics(ledger, data, costs, turnover)
    durations = data["durations"]
    weights = durations/durations.sum()
    exposure = (None if data["beta"] is None else data["held"][:-1]*data["beta"][None])
    total_beta = None if exposure is None else exposure.sum(axis=1)
    beta_signed = None if exposure is None else weights@exposure
    beta_abs_allocation = (None if exposure is None else
                           weights@(np.sign(total_beta)[:, None]*exposure))
    rows = []
    for i, label in enumerate(data["labels"]):
        rows.append(dict(
            asset=label, gross_pnl=float(gross[i]), gross_return=float(gross[i]/initial),
            arrival_slippage=float(arrival[:, i].sum()), commission=float(commissions[:, i].sum()),
            cost_pnl=float(cost_asset[i]), cost_return=float(cost_asset[i]/initial),
            net_pnl=float(gross[i]-cost_asset[i]), net_return=float((gross[i]-cost_asset[i])/initial),
            turnover=float(turnover[:, i].sum()),
            held_abs_weight_mean=float(weights@np.abs(data["held"][:-1, i])),
            beta_contribution_mean=None if beta_signed is None else float(beta_signed[i]),
            beta_abs_mean_contribution=None if beta_abs_allocation is None else float(beta_abs_allocation[i]),
            terminal_cost_return=float(diagnostics["terminal_cost"][i]/initial),
            terminal_turnover=float(diagnostics["terminal_turnover"][i]),
            boundary_cost_return=float(diagnostics["boundary_cost"][i]/initial),
            boundary_turnover=float(diagnostics["boundary_turnover"][i]),
            trade_sign_reversals=int(diagnostics["trade_sign_reversals"][i]),
            nonterminal_fill_events=int(diagnostics["nonterminal_fill_events"][i]),
            boundary_fill_events=int(diagnostics["boundary_fill_events"][i]),
        ))
    additive = ("gross_pnl", "gross_return", "arrival_slippage", "commission", "cost_pnl",
                "cost_return", "net_pnl", "net_return", "turnover", "terminal_cost_return",
                "terminal_turnover", "boundary_cost_return", "boundary_turnover",
                "trade_sign_reversals", "nonterminal_fill_events", "boundary_fill_events")
    totals = {key: sum(row[key] for row in rows) for key in additive}
    totals.update(beta_mean=None if total_beta is None else float(weights@total_beta),
                  beta_abs_mean=None if total_beta is None else float(weights@np.abs(total_beta)),
                  boundary_target_decisions=diagnostics["boundary_target_decisions"],
                  terminal_order_events=diagnostics["terminal_order_events"])
    return dict(schema_version=1, assets=data["labels"], initial_equity=initial,
                duration_seconds=float(durations.sum()), interval_count=len(durations),
                totals=totals, asset_rows=rows, reconciliation_max_abs_errors=errors,
                definitions=dict(
                    return_denominator="session initial equity",
                    exposure_basis="post-fill marked holdings at interval start, duration weighted",
                    beta_abs_mean_contribution="sign(total held beta) times each asset's signed beta contribution; can be negative",
                    boundary_basis="originating target is at the supplied beta bound; descriptive, not causal",
                    beta_bound=data["beta_bound"], boundary_tolerance=data["tolerance"],
                    reversal_trade_tolerance=diagnostics["numerical_trade_tolerance"],
                    terminal="closing fill costs included once; final flat row has no holding duration"))


def paired_execution_attribution(reference_ledger, candidate_ledger, prices, *, beta,
                                 beta_bound=.1, tolerance=1e-8):
    """Attribute candidate-minus-reference accounting and absolute-beta reduction.

    P&L, cost and turnover differences are additive per asset. Reduction in
    duration-weighted mean absolute beta uses the secant of ``abs`` between
    the two aggregate exposures on each interval. This reconciles exactly,
    including beta sign crossings; it is a descriptive symmetric allocation,
    not a causal estimate of an asset's effect or a risk-adjusted return.
    """
    reference = _prepare(reference_ledger, prices, beta, beta_bound, tolerance)
    candidate = _prepare(candidate_ledger, prices, beta, beta_bound, tolerance)
    if (not np.array_equal(reference["times"], candidate["times"])
            or not np.array_equal(reference["assets"], candidate["assets"])
            or not np.array_equal(reference["decisions"], candidate["decisions"])
            or reference["initial"] != candidate["initial"]):
        raise ValueError("paired ledgers must share clocks, assets and initial equity")
    before = execution_attribution(reference_ledger, prices, beta=beta,
                                   beta_bound=beta_bound, tolerance=tolerance)
    after = execution_attribution(candidate_ledger, prices, beta=beta,
                                  beta_bound=beta_bound, tolerance=tolerance)
    ref_contributions = reference["held"][:-1]*reference["beta"][None]
    new_contributions = candidate["held"][:-1]*candidate["beta"][None]
    ref_beta, new_beta = ref_contributions.sum(axis=1), new_contributions.sum(axis=1)
    difference = ref_beta-new_beta
    slope = np.divide(np.abs(ref_beta)-np.abs(new_beta), difference,
                      out=np.zeros_like(difference), where=difference != 0)
    interval_allocation = slope[:, None]*(ref_contributions-new_contributions)
    mean_allocation = reference["durations"]@interval_allocation/reference["durations"].sum()
    reduction = before["totals"]["beta_abs_mean"]-after["totals"]["beta_abs_mean"]
    allocation_error = _reconcile(mean_allocation.sum(), reduction, "absolute-beta reduction allocation")
    difference_keys = ("gross_pnl", "gross_return", "arrival_slippage", "commission", "cost_pnl",
                       "cost_return", "net_pnl", "net_return", "turnover", "terminal_cost_return",
                       "terminal_turnover", "boundary_cost_return", "boundary_turnover",
                       "trade_sign_reversals", "nonterminal_fill_events", "boundary_fill_events")
    rows = []
    for i, (old, new) in enumerate(zip(before["asset_rows"], after["asset_rows"])):
        row = {key:new[key]-old[key] for key in difference_keys}
        row.update(asset=old["asset"],
                   beta_abs_mean_reduction_contribution=float(mean_allocation[i]),
                   beta_contribution_mean_change=new["beta_contribution_mean"]-old["beta_contribution_mean"],
                   held_abs_weight_mean_change=new["held_abs_weight_mean"]-old["held_abs_weight_mean"])
        rows.append(row)
    totals = {key:after["totals"][key]-before["totals"][key] for key in difference_keys}
    totals["beta_abs_mean_reduction"] = float(reduction)
    return dict(schema_version=1, assets=reference["labels"],
                reference=before, candidate=after, totals=totals, asset_rows=rows,
                beta_allocation_max_abs_error=allocation_error,
                definitions=dict(
                    differences="candidate minus reference except beta_abs_mean_reduction",
                    beta_abs_mean_reduction="reference minus candidate",
                    beta_allocation="interval absolute-value secant times per-asset beta change; signed and descriptive, not causal"))
