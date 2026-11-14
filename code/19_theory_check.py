"""STEP 19 - Comparative statics of the entry model when the probability of winning depends on type (w > 0).

Section 3.3 of the paper first assumes that the winner is drawn at random among entrants (probability 1/N, independent of the
renegotiation technology rho_i). Under the Documentos Tipo the lowest value scores with probability 1/4, so the weight w on the
price-only optimum in the bid function is strictly positive and firms with a larger rho_i bid lower. This script relaxes the
equiprobability assumption and checks numerically the analytic results of Appendix 'Comparative statics' (app:theory):

  * entrant i wins with probability p_i = s(rho_i) / sum_k s(rho_k),  s(rho) = 1 + eta * rho,  eta = psi * w * R_j,
    where psi >= 0 maps the bid advantage that rho gives (w * R_j * rho) into a winning probability (psi = 0: equiprobability);
  * entry: p(rho*) (m + rho* R_j) = k, N_j = Nbar (1 - F(rho*));
  * winner's expected type M = E[rho s] / E[s] over entrants = mu + eta sigma^2 / (1 + eta mu).

  dM/dR = -h(rho*) theta* (M - rho*) |d rho*/dR|  +  psi w sigma^2 / (1 + eta mu)^2
          (entry threshold falls: type falls)       (sorting sharpens with the rent: type rises)

The script (i) verifies the decomposition against a finite-difference derivative of the solved equilibrium, (ii) verifies dN/dR > 0 and
d(M R)/dR > 0 on a grid, and (iii) tabulates the band of weights w in which dM/dR > 0 (sign reversal) as a function of the selectivity of
entry and of psi, for a uniform and a Beta(2,2) distribution of rho.

Outputs: tab_theory.tex, tables/theory_threshold.csv, numbers
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize

from config import TABLES
from tex import fnum, put_number, write_numbers_tex, write_table

M_ = 1.0                         # price margin m (all money amounts in units of m)
NBAR = 60.0                      # pool of potential entrants
R0 = 6.0                         # evaluation point for the table
PSI_GRID = (0.5, 2.0, 10.0, 50.0)
RHO0_GRID = (0.02, 0.05, 0.10, 0.25, 0.50)
W_SCAN = np.unique(np.concatenate([np.geomspace(1e-4, 0.05, 80), np.linspace(0.05, 1.0, 191)]))


class Dist:
    """rho in [0, 1] with a polynomial density sum_j c_j rho^j (closed-form truncated moments)."""

    def __init__(self, name: str, coefs: dict):
        self.name, self.c = name, coefs

    def mom(self, lo: float, k: int) -> float:
        return sum(c * (1 - lo ** (k + j + 1)) / (k + j + 1) for j, c in self.c.items())

    def pdf(self, x: float) -> float:
        return sum(c * x ** j for j, c in self.c.items())

    def sf(self, lo: float) -> float:
        return self.mom(lo, 0)

    def stats(self, lo: float):
        s = self.sf(lo)
        m1, m2 = self.mom(lo, 1) / s, self.mom(lo, 2) / s
        return m1, m2 - m1 ** 2, m2, self.pdf(lo) / s            # mean, variance, E[rho^2], hazard


def solve(dist: Dist, kn: float, R: float, psi: float, w: float):
    """Entry threshold rho*(R) with eta = psi*w*R. kn = k * Nbar. Returns (rho*, N, M, mu, sigma2, theta*, h, eta)."""
    eta = psi * w * R

    def phi(a):
        mu = dist.mom(a, 1) / dist.sf(a)
        return ((1 + eta * a) / (1 + eta * mu)) * (M_ + a * R) - kn * dist.sf(a)

    lo, hi = 1e-9, 1 - 1e-3
    a = lo if phi(lo) > 0 else hi if phi(hi) < 0 else optimize.brentq(phi, lo, hi, xtol=1e-14)
    mu, sig2, m2, h = dist.stats(a)
    s_bar = 1 + eta * mu
    return a, NBAR * dist.sf(a), (mu + eta * m2) / s_bar, mu, sig2, (1 + eta * a) / s_bar, h, eta


def dM_dR(dist: Dist, kn: float, R: float, psi: float, w: float, dR: float = 1e-5):
    up, dn = solve(dist, kn, R + dR, psi, w), solve(dist, kn, R - dR, psi, w)
    a, N, M, mu, sig2, theta, h, eta = solve(dist, kn, R, psi, w)
    fd = (up[2] - dn[2]) / (2 * dR)
    da = (up[0] - dn[0]) / (2 * dR)
    entry = -h * theta * (M - a) * abs(da)
    sorting = psi * w * sig2 / (1 + eta * mu) ** 2
    return dict(fd=fd, entry=entry, sorting=sorting, dN=(up[1] - dn[1]) / (2 * dR),
                dMR=(up[2] * (R + dR) - dn[2] * (R - dR)) / (2 * dR), rho_star=a, N=N, M=M)


def kn_for(dist: Dist, rho0: float, R: float = R0) -> float:
    """k*Nbar that makes the entry threshold equal rho0 at w = 0 (equiprobable winner)."""
    return (M_ + rho0 * R) / dist.sf(rho0)


def reversal_band(dist: Dist, kn: float, psi: float, R: float = R0):
    pos = np.array([dM_dR(dist, kn, R, psi, w)["fd"] > 0 for w in W_SCAN])
    if not pos.any():
        return None
    return float(W_SCAN[pos].min()), float(W_SCAN[pos].max())


def main() -> None:
    dists = [Dist("Uniform", {0: 1.0}), Dist("Beta(2,2)", {1: 6.0, 2: -6.0})]
    rows, max_gap, ok_dn, ok_dmr, ok_w0 = [], 0.0, True, True, True
    for dist in dists:
        for rho0 in RHO0_GRID:
            kn = kn_for(dist, rho0)
            for R in (2.0, 4.0, 6.0, 9.0, 12.0):
                for psi in (0.0,) + PSI_GRID:
                    for w in (0.0, 0.05, 0.25, 0.5, 1.0):
                        r = dM_dR(dist, kn, R, psi, w)
                        max_gap = max(max_gap, abs(r["fd"] - r["entry"] - r["sorting"]))
                        ok_dn &= r["dN"] > 0
                        ok_dmr &= (r["dMR"] > 0) or (r["fd"] > 0)
                        ok_w0 &= (w > 0) or (r["fd"] < 0)
            for psi in PSI_GRID:
                band = reversal_band(dist, kn, psi)
                rows.append(dict(dist=dist.name, rho0=rho0, psi=psi, kn=kn, w_lo=band[0] if band else np.nan,
                                 w_hi=band[1] if band else np.nan))
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "theory_threshold.csv", index=False)
    assert ok_dn, "dN/dR must be positive"
    assert ok_w0, "with w = 0 the winner's expected type must fall with the rent"
    assert ok_dmr, "d(M R)/dR must be positive"
    assert max_gap < 1e-6, f"analytic decomposition disagrees with finite differences by {max_gap:.2e}"
    mant, ex = f"{max_gap:.1e}".split("e")
    put_number("theory max gap", "$" + mant + "\\times10^{" + str(int(ex)) + "}$")
    put_number("theory share reversal", 100 * df.w_lo.notna().mean(), ".0f")
    # smallest rho0 for which no reversal exists at any psi in the grid
    safe = df.groupby(["dist", "rho0"]).w_lo.apply(lambda s: s.isna().all()).reset_index()
    put_number("theory safe rho0", safe[safe.w_lo].rho0.min(), ".2")
    put_number("theory reversal rho0", df.loc[df.w_lo.notna(), "rho0"].max(), ".2")
    put_number("theory psi max", max(PSI_GRID), ".0")

    body = []
    for dist in dists:
        body.append(f"\\multicolumn{{5}}{{l}}{{\\emph{{{dist.name}}}}} \\\\")
        for rho0 in RHO0_GRID:
            sub = df[(df.dist == dist.name) & (df.rho0 == rho0)].sort_values("psi")
            cells = [("none" if np.isnan(r.w_lo) else f"({fnum(r.w_lo, 2)}, {fnum(r.w_hi, 2)})") for r in sub.itertuples()]
            body.append(f"$\\rho^*_0={rho0:.2f}$ & " + " & ".join(cells) + " \\\\")
    write_table("tab_theory", "\n".join(body), "", "tab:theory",
                "Band of weights $w$ (weight of the price-only optimum in the bid, \\cref{eq:bid}) for which the winner's expected renegotiation type \\emph{rises} with the project rent, $dE[\\rho_w]/dR_j>0$; "
                "``none'' means $dE[\\rho_w]/dR_j<0$ for every $w\\in[0,1]$. Rows: selectivity of entry, the entry threshold $\\rho^*_0$ that obtains with a randomly drawn winner at $R_j=6$ (a low $\\rho^*_0$ means that almost every firm enters). "
                "Columns: $\\psi$, which converts the bid advantage $wR_j\\rho_i$ into a winning probability, $p_i=s(\\rho_i)/\\sum_k s(\\rho_k)$ with $s(\\rho)=1+\\psi wR_j\\rho$. "
                "Stylised parameters, money in units of the price margin ($m=1$), $\\bar N=60$; the entry cost is set so that the threshold equals $\\rho^*_0$. "
                "On the whole grid of the script (five values of $R_j$ from 2 to 12, all rows and columns) $dN_j/dR_j>0$ and $d(E[\\rho_w]R_j)/dR_j>0$, and the decomposition of \\cref{eq:dMdR} matches finite differences.",
                "lcccc", "& $\\psi=0.5$ & $\\psi=2$ & $\\psi=10$ & $\\psi=50$")
    write_numbers_tex()
    print(df.to_string())
    print("max |finite-difference - analytic| =", max_gap)


if __name__ == "__main__":
    main()
