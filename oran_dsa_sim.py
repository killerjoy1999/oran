#!/usr/bin/env python3
"""
Simulation of: "Interoperable rApp/xApp Control over O-RAN for Mobility-aware
Dynamic Spectrum Allocation" (Giannopoulos et al., arXiv:2601.13769).

Stages
  lstm  : rApp traffic predictor (LSTM, numpy)  -> Fig. 4(a), 4(b)
  fig5  : xApp success rate / Jain index vs numerology & demand -> Fig. 5
  fig6  : policy comparison (coloring + fairness)               -> Fig. 6
  day   : closed-loop rApp->xApp run over the test day (extra)  -> Fig. 7
Usage:  python oran_dsa_sim.py --stage all [--quick]
"""
import argparse, json, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ Table I parameters ============================
B, BG = 10e6, 0.25e6                       # total bandwidth, one-sided guard band
RU_POS = np.array([[0, 0], [200, 0], [-200, 0]], float)   # macro, micro, micro
RU_RAD = np.array([300., 50., 50.])
RU_PW = np.array([0.1, 0.01, 0.01])        # per-PRB Tx power [W]
RU_ALPHA = np.array([2.7, 2.8, 2.8])       # path-loss exponents
K_PL = 1.0
P_MIN = 1e-3
NOISE_DENS = 10 ** (-174 / 10) * 1e-3      # W/Hz
SPEED, DT = 1.5, 1.0                       # m/s, s
EWMA_ALPHA = 0.1
NR = 3


def n_prb(mu):                             # Eq. (2)
    return int(np.floor((B - 2 * BG) / (180e3 * 2 ** mu)))


def prb_bw(mu):                            # W = 12 * 15kHz * 2^mu
    return 12 * 15e3 * 2 ** mu


# ============================ Traffic dataset (synthetic) ===================
# The public 5G dataset [21] is not reachable offline, so a statistically
# similar diurnal trace (15-min samples, 6 months) is synthesised.
def gen_traffic(days=180, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(96) / 4.0
    sg = lambda x: 1 / (1 + np.exp(-x))
    prof = (1.00 * (sg((h - 5.5) / 0.35) - sg((h - 9.5) / 0.45))
            + 0.95 * (sg((h - 14.0) / 0.45) - sg((h - 19.5) / 0.55)))
    base, amp = 7.0, 150.0
    out = []
    for d in range(days):
        scale = rng.uniform(0.92, 1.03) * (0.85 if d % 7 in (5, 6) else 1.0)
        if d == days - 1:
            scale = 1.0                     # test day: clean weekday
        noise = 1 + 0.035 * rng.standard_normal(96)
        out.append((base + amp * prof) * scale * noise)
    lam = np.clip(np.concatenate(out), 6, 160)
    return lam


def mu_from_load(l):                       # capacity-based decision regions
    return 4 if l <= 9 else 3 if l <= 18 else 2 if l <= 39 else 1 if l <= 78 else 0


# ============================ Numpy LSTM (rApp) =============================
def sig_(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))


class LSTM:
    def __init__(self, H=16, seed=0):
        r = np.random.default_rng(seed)
        self.H = H
        self.p = {"Wx": r.normal(0, 0.5, (1, 4 * H)),
                  "Wh": r.normal(0, 1 / np.sqrt(H), (H, 4 * H)),
                  "b": np.zeros(4 * H), "Wy": r.normal(0, 0.3, (H, 1)), "by": np.zeros(1)}
        self.p["b"][H:2 * H] = 1.0
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def forward(self, X, cache=False):
        p, H = self.p, self.H
        N, L = X.shape
        h = np.zeros((N, H)); c = np.zeros((N, H)); cs = []
        for t in range(L):
            x = X[:, t:t + 1]
            z = x @ p["Wx"] + h @ p["Wh"] + p["b"]
            i = sig_(z[:, :H]); f = sig_(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H]); o = sig_(z[:, 3 * H:])
            cn = f * c + i * g; tc = np.tanh(cn); hn = o * tc
            if cache: cs.append((x, h, c, i, f, g, o, tc))
            h, c = hn, cn
        return h @ p["Wy"] + p["by"], h, cs

    def step(self, X, Y, lr):
        p, H = self.p, self.H
        y, h, cs = self.forward(X, True)
        N = len(X)
        dy = 2 * (y - Y) / N
        gr = {k: np.zeros_like(v) for k, v in p.items()}
        gr["Wy"] = h.T @ dy; gr["by"] = dy.sum(0)
        dh = dy @ p["Wy"].T; dc = np.zeros_like(dh)
        for t in reversed(range(len(cs))):
            x, hp, cp, i, f, g, o, tc = cs[t]
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            dz = np.concatenate([dc * g * i * (1 - i), dc * cp * f * (1 - f),
                                 dc * i * (1 - g ** 2), do * o * (1 - o)], 1)
            gr["Wx"] += x.T @ dz; gr["Wh"] += hp.T @ dz; gr["b"] += dz.sum(0)
            dh = dz @ p["Wh"].T; dc = dc * f
        nrm = np.sqrt(sum((g ** 2).sum() for g in gr.values()))
        if nrm > 1.0:
            for k in gr: gr[k] /= nrm
        self.t += 1
        for k in p:                                    # Adam
            self.m[k] = 0.9 * self.m[k] + 0.1 * gr[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * gr[k] ** 2
            mh = self.m[k] / (1 - 0.9 ** self.t); vh = self.v[k] / (1 - 0.999 ** self.t)
            p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)


def windows(s, idx, L):
    return np.stack([s[i - L:i] for i in idx]), s[idx][:, None]


def train_eval(s, n_train, L, lr, epochs, nwin, seed=0, batch=64):
    rng = np.random.default_rng(seed)
    net = LSTM(16, seed)
    valid = np.arange(L, n_train)
    for _ in range(epochs):
        idx = rng.choice(valid, min(nwin, len(valid)), replace=False)
        for b in range(0, len(idx), batch):
            X, Y = windows(s, idx[b:b + batch], L)
            net.step(X, Y, lr)
    tidx = np.arange(n_train, len(s))
    X, Y = windows(s, tidx, L)
    pred = net.forward(X)[0]
    mse = float(np.mean((pred - Y) ** 2))
    if not np.isfinite(mse): mse, pred = 1.0, np.zeros_like(Y)
    return mse, pred[:, 0]


def stage_lstm(a):
    lam = gen_traffic()
    s = lam / 160.0
    n_train = 179 * 96
    Ls = {"1 hour": 4, "12 hours": 48, "1 day": 96, "1 week": 672}
    lrs = [0.1, 0.01, 0.001]
    nwin = {4: 3000, 48: 2500, 96: 1800, 672: 640}
    ep = 3 if a.quick else 8
    res, best, bestpred = {}, 9, None
    for name, L in Ls.items():
        for lr in lrs:
            t0 = time.time()
            mse, pred = train_eval(s, n_train, L, lr, ep, nwin[L])
            res[f"{name}|{lr}"] = mse
            print(f"  L={name:9s} lr={lr:<6} MSE={mse:.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if L == 48 and lr == 0.01:
                bestpred = pred
    # Fig 4(a)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    w = 0.25
    for j, lr in enumerate(lrs):
        ax[0].bar(np.arange(4) + (j - 1) * w, [res[f"{n}|{lr}"] for n in Ls], w, label=f"α = {lr}")
    ax[0].set_yscale("log"); ax[0].set_xticks(range(4)); ax[0].set_xticklabels(list(Ls))
    ax[0].set_xlabel("Lookback window L"); ax[0].set_ylabel("Converged MSE (test day)")
    ax[0].legend(); ax[0].set_title("(a)")
    # Fig 4(b)
    hrs = np.arange(96) / 4
    act = lam[n_train:]; prd = np.clip(bestpred * 160, 0, 200)
    ax[1].plot(hrs, act, "k-", label="Actual"); ax[1].plot(hrs, prd, "b:", label="Predicted")
    for th, mu in zip([9, 18, 39, 78], [4, 3, 2, 1]):
        ax[1].axhline(th, color="r", ls="--", lw=0.8)
    for th, mu in zip([9, 18, 39, 78, 130], [4, 3, 2, 1, 0]):
        ax[1].text(23.9, th + 2, f"μ={mu}", color="r", ha="right", fontsize=8)
    ax[1].set_xlabel("Time (hours)"); ax[1].set_ylabel("Traffic load (# devices)")
    ax[1].set_xlim(0, 24); ax[1].legend(loc="upper left"); ax[1].set_title("(b)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig4_rapp_training_inference.png", dpi=160); plt.close()
    json.dump({"mse": res, "actual": act.tolist(), "pred": prd.tolist()},
              open(f"{a.out}/lstm_results.json", "w"))


# ============================ xApp simulator ================================
def color_greedy(adj, P, order):
    U = len(adj); c = -np.ones(U, int)
    for u in order:
        nb = c[adj[u]]; nb = nb[nb >= 0]
        used = np.zeros(P, bool); used[nb] = True
        f = int(np.argmin(used))
        if not used[f]: c[u] = f
    return c


def dsatur(adj, P):
    U = len(adj); c = -np.ones(U, int); deg = adj.sum(1)
    sat = np.zeros(U, int); sets = [set() for _ in range(U)]
    unc = np.ones(U, bool)
    for _ in range(U):
        cand = np.flatnonzero(unc)
        u = cand[np.lexsort((-deg[cand], -sat[cand]))[0]]
        unc[u] = False
        nb = c[adj[u]]; nb = nb[nb >= 0]
        used = np.zeros(P, bool); used[nb] = True
        f = int(np.argmin(used))
        if not used[f]:
            c[u] = f
            for v in np.flatnonzero(adj[u] & unc):
                if f not in sets[v]: sets[v].add(f); sat[v] += 1
    return c


class Sim:
    def __init__(self, U, mu, demand, seed, eta=0.0, alpha=EWMA_ALPHA):
        self.rng = np.random.default_rng(seed)
        self.U, self.mu, self.P = U, mu, n_prb(mu)
        self.W = prb_bw(mu); self.N = NOISE_DENS * self.W
        self.ru = (np.arange(U) + self.rng.integers(0, NR)) % NR
        rad = RU_RAD[self.ru] * np.sqrt(self.rng.random(U))
        ang = self.rng.uniform(0, 2 * np.pi, U)
        self.pos = RU_POS[self.ru] + rad[:, None] * np.c_[np.cos(ang), np.sin(ang)]
        self.th = self.rng.uniform(0, 2 * np.pi, U)
        self.d = np.full(U, float(demand)) if np.isscalar(demand) else self.rng.choice(demand, U)
        self.w = np.ones(U); self.eta = np.full(U, eta); self.alpha = alpha
        self.Rbar = self.d.copy(); self.t = 0
        self.S = np.zeros(U); self.A = np.zeros(U); self.unc_sum = 0

    # --- mobility (Eq. 3, correlated random walk, confined to home cell) ----
    def move(self):
        self.th += self.rng.normal(0, 0.35, self.U)
        self.pos += SPEED * DT * np.c_[np.cos(self.th), np.sin(self.th)]
        c = RU_POS[self.ru]; v = self.pos - c
        dist = np.linalg.norm(v, axis=1); out = dist > RU_RAD[self.ru]
        if out.any():
            self.th[out] = np.arctan2(-v[out, 1], -v[out, 0]) + self.rng.uniform(-.5, .5, out.sum())
            self.pos[out] = c[out] + v[out] * (0.98 * RU_RAD[self.ru][out] / dist[out])[:, None]

    # --- channel (Eqs. 4-5) -------------------------------------------------
    def channel(self):
        D = np.maximum(np.linalg.norm(self.pos[None] - RU_POS[:, None], axis=2), 1.0)
        self.G = K_PL * D ** (-RU_ALPHA[:, None]) * self.rng.exponential(1.0, (NR, self.U))
        self.sig = RU_PW[self.ru] * self.G[self.ru, np.arange(self.U)]

    # --- Interference Graph Engine -----------------------------------------
    def build_graph(self):
        ru = self.ru
        IM = self.G.T[:, ru] * RU_PW[ru][None, :]                # IM[u,v]: RU(v) -> UE u
        Rm = self.W * np.log2(1 + self.sig[:, None] / (IM + self.N)) / 1e6
        Cm = Rm < ((1 - self.eta) * self.d)[:, None]
        adj = Cm | Cm.T | (ru[:, None] == ru[None, :])
        np.fill_diagonal(adj, False)
        return adj

    def rate_on(self, u, p, occ):
        I = self.N
        for k in range(NR):
            if k != self.ru[u] and occ[k, p] >= 0: I += RU_PW[k] * self.G[k, u]
        return self.W * np.log2(1 + self.sig[u] / I) / 1e6

    # --- Fairness Engine (Alg. 2: MPF; RR variant) ---------------------------
    def fairness(self, occ, adj, mode):
        U, P = self.U, self.P
        sched = np.zeros(U, bool); sched[occ[occ >= 0]] = True
        unc = np.flatnonzero(~sched)
        self.unc_sum += len(unc)
        if len(unc) == 0: return
        rank = (np.arange(U) - self.t) % U
        order = unc[np.argsort(self.Rbar[unc] / self.w[unc])] if mode == "MPF" \
            else unc[np.argsort(rank[unc])]
        for u in order:
            r = self.ru[u]
            ok = np.ones(P, bool); I = np.full(P, self.N)
            for k in range(NR):
                if k == r: continue
                o = occ[k]; has = o >= 0
                if has.any():
                    conf = np.zeros(P, bool); conf[has] = adj[u, o[has]]
                    ok &= ~conf
                    I = I + has * RU_PW[k] * self.G[k, u]
            if not ok.any(): continue
            rate = self.W * np.log2(1 + self.sig[u] / I) / 1e6
            m = np.where(ok, rate / self.Rbar[u] * self.w[u] if mode == "MPF" else rate, -np.inf)
            p = int(np.argmax(m)); v = occ[r, p]
            if v >= 0:
                if mode == "MPF":
                    mv = self.rate_on(v, p, occ) / self.Rbar[v] * self.w[v]
                    if not m[p] > mv: continue
                elif not rank[u] < rank[v]: continue
            occ[r, p] = u

    # --- one xApp scheduling interval ---------------------------------------
    def step(self, coloring="WP", fair="MPF"):
        self.move(); self.channel()
        U, P = self.U, self.P
        occ = -np.ones((NR, P), int); adj = None
        if coloring in ("WP", "G", "DSatur"):
            adj = self.build_graph()
            if coloring == "WP":
                c = color_greedy(adj, P, np.argsort(-(self.w * adj.sum(1)), kind="stable"))
            elif coloring == "G":
                c = color_greedy(adj, P, np.arange(U))
            else:
                c = dsatur(adj, P)
            col = np.flatnonzero(c >= 0); occ[self.ru[col], c[col]] = col
            if fair in ("MPF", "RR"): self.fairness(occ, adj, fair)
        elif coloring == "Random":
            for r in range(NR):
                m = np.flatnonzero(self.ru == r)
                if len(m): occ[r] = self.rng.choice(m, P)
        elif coloring == "Seq":
            for r in range(NR):
                m = np.flatnonzero(self.ru == r)[:P]; occ[r, :len(m)] = m
        # ---- evaluation with the true co-channel interference (Eq. 6-7)
        Ru = np.zeros(U)
        for r in range(NR):
            has = occ[r] >= 0
            if not has.any(): continue
            pi = np.flatnonzero(has); u = occ[r][pi]
            I = np.full(len(u), self.N)
            for k in range(NR):
                if k != r: I = I + (occ[k][pi] >= 0) * RU_PW[k] * self.G[k, u]
            np.add.at(Ru, u, self.W * np.log2(1 + RU_PW[r] * self.G[r, u] / I) / 1e6)
        sat = Ru >= (1 - self.eta) * self.d
        asg = np.zeros(U, bool); asg[occ[occ >= 0]] = True
        self.Rbar = np.maximum((1 - self.alpha) * self.Rbar + self.alpha * Ru, 1e-3)
        self.S += sat; self.A += asg; self.t += 1

    def summary(self):
        Sb = self.S / self.t
        jfi = 100 * Sb.sum() ** 2 / (self.U * (Sb ** 2).sum()) if Sb.sum() > 0 else 0.0
        return dict(success=100 * Sb.mean(), assigned=100 * (self.A / self.t).mean(),
                    jfi=jfi, unc=self.unc_sum / self.t)


def run(mu, U, d, Nx, seed, coloring="WP", fair="MPF"):
    s = Sim(U, mu, d, seed)
    for _ in range(Nx): s.step(coloring, fair)
    return s.summary()


# ============================ Fig. 5 ========================================
REGIONS = {0: (79, 160), 1: (40, 78), 2: (19, 39), 3: (10, 18), 4: (6, 9)}


def stage_fig5(a):
    Nx = 100 if a.quick else a.nx
    nd = 2 if a.quick else a.draws
    demands = [0.5, 1.0, 1.5]
    res = {d: {"success": [], "jfi": []} for d in demands}
    for d in demands:
        for mu in range(5):
            lo, hi = REGIONS[mu]
            Us = np.unique(np.round(np.linspace(lo, hi, nd)).astype(int))
            outs = [run(mu, int(U), d, Nx, seed=100 * mu + i) for i, U in enumerate(Us)]
            res[d]["success"].append(float(np.mean([o["success"] for o in outs])))
            res[d]["jfi"].append(float(np.mean([o["jfi"] for o in outs])))
            print(f"  d={d} mu={mu} U={list(Us)} succ={res[d]['success'][-1]:.1f} "
                  f"jfi={res[d]['jfi'][-1]:.1f}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for d, mk in zip(demands, "osv"):
        ax[0].plot(range(5), res[d]["success"], marker=mk, label=f"d_u = {d} Mbps")
        ax[1].plot(range(5), res[d]["jfi"], marker=mk, label=f"d_u = {d} Mbps")
    ax[0].set_ylabel("Success Rate (%)"); ax[1].set_ylabel("Jain's Fairness Service-Share (%)")
    for x, t in zip(ax, ["(a)", "(b)"]):
        x.set_xlabel("Numerology index μ"); x.set_xticks(range(5)); x.grid(alpha=.3)
        x.legend(); x.set_title(t)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig5_xapp_vs_numerology.png", dpi=160); plt.close()
    json.dump({str(k): v for k, v in res.items()}, open(f"{a.out}/fig5_results.json", "w"))


# ============================ Fig. 6 ========================================
def stage_fig6(a):
    Nx = 300 if a.quick else 900
    seeds = 2 if a.quick else 4
    algs = [("Random", "Random"), ("Seq-Color", "Seq"), ("G-Color", "G"),
            ("DSatur", "DSatur"), ("WP-Color", "WP")]
    A_sat, A_asg = [], []
    for nm, al in algs:
        o = [run(4, 9, 1.0, Nx, s, al, "MPF") for s in range(seeds)]
        A_sat.append(np.mean([x["success"] for x in o])); A_asg.append(np.mean([x["assigned"] for x in o]))
        print(f"  {nm:10s} assigned={A_asg[-1]:.1f} satisfied={A_sat[-1]:.1f}", flush=True)
    fl = [("No Fairness", "none"), ("RR", "RR"), ("MPF", "MPF")]
    J = []
    for nm, f in fl:
        o = [run(4, 14, 1.0, Nx, s, "WP", f) for s in range(seeds)]
        J.append(np.mean([x["jfi"] for x in o]))
        print(f"  {nm:12s} JFI={J[-1]:.1f}", flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(5); w = 0.38
    b1 = ax[0].bar(x - w / 2, A_asg, w, label="UEs assigned a PRB", color="#1f3a6b")
    b2 = ax[0].bar(x + w / 2, A_sat, w, label="UEs fully satisfied", color="#6aa6d8")
    ax[0].set_xticks(x); ax[0].set_xticklabels([n for n, _ in algs])
    ax[0].set_xlabel("PRB Allocation Policies (π4)"); ax[0].set_ylabel("Success Rate (%)")
    ax[0].set_ylim(0, 108); ax[0].legend(loc="lower right"); ax[0].set_title("(a)")
    for b in list(b1) + list(b2):
        ax[0].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}",
                   ha="center", fontsize=7)
    bb = ax[1].bar([n for n, _ in fl], J, color="#4fb8ea")
    for b in bb: ax[1].text(b.get_x() + b.get_width() / 2, b.get_height() + 1, f"{b.get_height():.1f}", ha="center", fontsize=8)
    ax[1].set_ylim(0, 108); ax[1].set_xlabel("Fairness Policies (π2)")
    ax[1].set_ylabel("Jain's Fairness Service-Share (%)"); ax[1].set_title("(b)")
    plt.tight_layout(); plt.savefig(f"{a.out}/fig6_policy_comparison.png", dpi=160); plt.close()
    json.dump({"algs": [n for n, _ in algs], "assigned": A_asg, "satisfied": A_sat,
               "fair": [n for n, _ in fl], "jfi": J}, open(f"{a.out}/fig6_results.json", "w"))


# ============================ Closed-loop day (rApp -> xApp) ================
def stage_day(a):
    lam = gen_traffic()[179 * 96:]
    pf = f"{a.out}/lstm_results.json"
    pred = np.array(json.load(open(pf))["pred"]) if os.path.exists(pf) else lam
    nslot = 6 if a.quick else 15
    cfgs = {"rApp-adaptive μ": None, "fixed μ=0": 0, "fixed μ=2": 2, "fixed μ=4": 4}
    succ = {k: [] for k in cfgs}; mus = []
    for i in range(96):
        U = int(np.clip(round(lam[i]), 6, 160)); mu_ad = mu_from_load(pred[i]); mus.append(mu_ad)
        for k, m in cfgs.items():
            o = run(mu_ad if m is None else m, U, 1.0, nslot, seed=i)
            succ[k].append(o["success"])
        if i % 12 == 0: print(f"  interval {i}/96", flush=True)
    hrs = np.arange(96) / 4
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for k in cfgs: ax[0].plot(hrs, succ[k], label=f"{k} (mean {np.mean(succ[k]):.1f}%)")
    ax[0].set_ylabel("Success rate (%)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].step(hrs, mus, where="post", color="r"); ax[1].set_ylabel("rApp-selected μ")
    ax[1].set_xlabel("Time of day (h)"); ax[1].set_yticks(range(5)); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{a.out}/fig7_closed_loop_day.png", dpi=160); plt.close()
    json.dump({k: float(np.mean(v)) for k, v in succ.items()}, open(f"{a.out}/fig7_results.json", "w"))
    print({k: round(float(np.mean(v)), 1) for k, v in succ.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "lstm", "fig5", "fig6", "day"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--nx", type=int, default=300, help="xApp slots per config in Fig.5")
    ap.add_argument("--draws", type=int, default=3, help="UE-count draws per numerology region")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("PRBs per numerology:", {m: n_prb(m) for m in range(5)})
    for name, fn in [("lstm", stage_lstm), ("fig5", stage_fig5), ("fig6", stage_fig6), ("day", stage_day)]:
        if a.stage in ("all", name):
            t0 = time.time(); print(f"== {name} ==", flush=True); fn(a)
            print(f"   done in {time.time()-t0:.0f}s", flush=True)
