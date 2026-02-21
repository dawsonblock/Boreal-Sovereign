import numpy as np
import hashlib
from dataclasses import dataclass

try:
    from numba import njit

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator


try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# Strict Cryptographic Determinism
np.random.seed(42)


# =====================================================================
# 1. BARE-METAL ALU MACROS (Hardened Q16.16 Fixed Point)
# =====================================================================
SHIFT = 16
ONE = 1 << 16
HALF = 1 << 15


@njit(cache=True)
def _alu_mul(a_q, b_q):
    return ((a_q * b_q) + HALF) >> SHIFT


@njit(cache=True)
def _alu_mac(w_q, z_q):
    # Manual loop to avoid Numba BLAS int64 unsupported error
    out = np.zeros(w_q.shape[0], dtype=np.int64)
    for i in range(w_q.shape[0]):
        sum_val = np.int64(0)
        for j in range(w_q.shape[1]):
            sum_val += w_q[i, j] * z_q[j]
        out[i] = (sum_val + HALF) >> SHIFT
    return out


@njit(cache=True)
def _alu_outer(e_q, z_q):
    out = np.zeros((e_q.shape[0], z_q.shape[0]), dtype=np.int64)
    for i in range(e_q.shape[0]):
        for j in range(z_q.shape[0]):
            out[i, j] = (e_q[i] * z_q[j] + HALF) >> SHIFT
    return out


@njit(cache=True)
def _alu_norm_sq(v_q):
    sum_val = np.int64(0)
    for i in range(v_q.shape[0]):
        sum_val += (v_q[i] * v_q[i]) >> SHIFT
    return sum_val


@njit(cache=True)
def _xorshift32(state):
    x = state & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0xFFFFFFFF


class ALU_Q16:
    SHIFT = SHIFT
    ONE = ONE
    HALF = HALF
    MAX_STATE = 15 * ONE  # Hardware Saturation Limits
    MAX_WEIGHT = 10 * ONE

    @classmethod
    def to_q(cls, x):
        return np.round(np.array(x, dtype=np.float64) * cls.ONE).astype(np.int64)

    @classmethod
    def to_f(cls, q):
        return np.array(q, dtype=np.float64) / cls.ONE

    @classmethod
    def sat(cls, x, limit):
        return np.clip(x, -limit, limit)

    @classmethod
    def mul(cls, a_q, b_q):
        return _alu_mul(a_q.astype(np.int64), b_q.astype(np.int64))

    @classmethod
    def mac(cls, W_q, z_q):
        return _alu_mac(W_q.astype(np.int64), z_q.astype(np.int64))

    @classmethod
    def outer(cls, e_q, z_q):
        return _alu_outer(e_q.astype(np.int64), z_q.astype(np.int64))

    @classmethod
    def norm_sq(cls, v_q):
        return _alu_norm_sq(v_q.astype(np.int64))

    @classmethod
    def mean(cls, arrays_q):
        """Pure integer mean across an ensemble to avoid float casting."""
        s = sum(a.astype(np.int64) for a in arrays_q)
        return (s + (len(arrays_q) >> 1)) // len(arrays_q)

    @classmethod
    def sqrt(cls, n_q):
        """Hardware-friendly integer square root for CEM Planner standard deviations."""
        if isinstance(n_q, np.ndarray):
            out = np.zeros_like(n_q, dtype=np.int64)
            for i in range(n_q.size):
                out.flat[i] = cls._sqrt_scalar(n_q.flat[i])
            return out
        return cls._sqrt_scalar(n_q)

    @classmethod
    def _sqrt_scalar(cls, n_q):
        if n_q <= 0:
            return np.int64(0)
        val = int(n_q) << cls.SHIFT
        x = val
        y = (x + 1) >> 1
        while y < x:
            x = y
            y = (x + val // x) >> 1
        return np.int64(x)


# =====================================================================
# 2. CONFIGURATION ARCHITECTURE
# =====================================================================
@dataclass
class ApexConfig:
    s_dim: int = 12
    s2_dim: int = 12
    o_dim: int = 24
    a_dim: int = 2
    n_cores: int = 3
    base_lr_shift: int = 5
    recov_lr_shift: int = 4  # Boosted learning rate for fault adaptation
    k_shift: int = 3
    decay_shift: int = 10  # 1023/1024 decay for infinite stability
    cem_iters: int = 2
    cem_pop: int = 15
    horizon: int = 3


config = ApexConfig()


# =====================================================================
# 3. Q16 PREDICTIVE CORE WITH SATURATION & DECAY
# =====================================================================
class Q16PredictiveCore:
    def __init__(self, seed_offset=0):
        np.random.seed(42 + seed_offset)
        # BOREAL RTL explicitly initializes to a deterministic Linear Hash + Identity
        total_w = (
            config.s_dim * config.s_dim
            + config.s_dim * config.a_dim
            + config.s_dim * config.s2_dim
            + config.o_dim * config.s_dim
        )
        w_flat = np.array(
            [
                (((i * 1664525) + seed_offset * 1013904223) % 16384) - 8192
                for i in range(total_w)
            ],
            dtype=np.int64,
        )

        offset = 0
        W11_flat = w_flat[offset : offset + config.s_dim * config.s_dim] >> 3

        offset += config.s_dim * config.s_dim
        W1a_flat = w_flat[offset : offset + config.s_dim * config.a_dim]

        offset += config.s_dim * config.a_dim
        W12_flat = w_flat[offset : offset + config.s_dim * config.s2_dim] >> 3

        offset += config.s_dim * config.s2_dim
        C_flat = w_flat[offset : offset + config.o_dim * config.s_dim]

        self.W11 = W11_flat.reshape((config.s_dim, config.s_dim))
        self.W1a = W1a_flat.reshape((config.s_dim, config.a_dim))
        self.W12 = W12_flat.reshape((config.s_dim, config.s2_dim))
        self.C = C_flat.reshape((config.o_dim, config.s_dim))

        # Apply Identity leaky overlays identical to `BOREAL` reset block
        for i in range(config.s_dim):
            self.W11[i, i] = int(0.9 * ALU_Q16.ONE)
        for i in range(min(config.s_dim, config.s2_dim)):
            self.W12[i, i] = int(0.1 * ALU_Q16.ONE)

    def step(self, s1_q, s2_q, a_q, o_q, active_lr_shift):
        # 2-Layer Hierarchical Top-Down Prediction
        s1_pred_q = (
            ALU_Q16.mac(self.W11, s1_q)
            + ALU_Q16.mac(self.W12, s2_q)
            + ALU_Q16.mac(self.W1a, a_q)
        )
        s1_pred_q = ALU_Q16.sat(s1_pred_q, ALU_Q16.MAX_STATE)

        o_pred_q = ALU_Q16.mac(self.C, s1_pred_q)

        # Bottom-Up Expectation Errors
        e_o_q = ALU_Q16.sat(o_q - o_pred_q, ALU_Q16.MAX_STATE)
        e_1_q = ALU_Q16.sat(s1_q - s1_pred_q, ALU_Q16.MAX_STATE)

        # State Correction (Local In-Place Error Flow)
        correction_1_q = ALU_Q16.mac(self.C.T, e_o_q) >> config.k_shift
        s1_new_q = ALU_Q16.sat(
            s1_pred_q + correction_1_q - (e_1_q >> 2), ALU_Q16.MAX_STATE
        )

        correction_2_q = ALU_Q16.mac(self.W12.T, e_1_q) >> config.k_shift
        s2_new_q = ALU_Q16.sat(s2_q + (correction_2_q >> 2), ALU_Q16.MAX_STATE)

        # In-Place Hebbian Hardware Updates
        self.C = ALU_Q16.sat(
            self.C + (ALU_Q16.outer(e_o_q, s1_pred_q) >> active_lr_shift),
            ALU_Q16.MAX_WEIGHT,
        )
        self.W11 = ALU_Q16.sat(
            self.W11 + (ALU_Q16.outer(e_1_q, s1_q) >> active_lr_shift),
            ALU_Q16.MAX_WEIGHT,
        )
        self.W12 = ALU_Q16.sat(
            self.W12 + (ALU_Q16.outer(e_1_q, s2_q) >> active_lr_shift),
            ALU_Q16.MAX_WEIGHT,
        )
        self.W1a = ALU_Q16.sat(
            self.W1a + (ALU_Q16.outer(e_1_q, a_q) >> active_lr_shift),
            ALU_Q16.MAX_WEIGHT,
        )

        # Hardware Weight Decay (Ensures eternal stability over millions of ticks)
        self.W11 -= self.W11 >> config.decay_shift
        self.W12 -= self.W12 >> config.decay_shift
        self.W1a -= self.W1a >> config.decay_shift
        self.C -= self.C >> config.decay_shift

        return s1_new_q, s2_new_q, e_o_q, o_pred_q


# =====================================================================
# 4. EPISTEMIC ENSEMBLE & L3 REGIME MEMORY
# =====================================================================
class MetaEpistemicEnsemble:
    def __init__(self):
        self.cores = [Q16PredictiveCore(i) for i in range(config.n_cores)]
        self.slow_z_q = np.zeros(config.s_dim, dtype=np.int64)
        self.regime_hash = "INIT"
        self.regime_stability = 0
        self.surprise_lp = 0

    def step(self, states1_q, states2_q, a_q, o_q, active_lr_shift):
        new_states1_q, new_states2_q, preds_q = [], [], []

        for i, core in enumerate(self.cores):
            s1_new, s2_new, e_o_q, o_pred = core.step(
                states1_q[i], states2_q[i], a_q, o_q, active_lr_shift
            )
            new_states1_q.append(s1_new)
            new_states2_q.append(s2_new)
            preds_q.append(o_pred)

        # True Integer Mean & Variance for Uncertainty
        mean_pred_q = ALU_Q16.mean(preds_q)
        unc_sq_q = (
            sum(ALU_Q16.norm_sq(p - mean_pred_q) for p in preds_q) // config.n_cores
        )

        # Calculate ensemble surprise based on the mean ensemble error, terminating random core bias
        mean_err_q = ALU_Q16.sat(o_q - mean_pred_q, ALU_Q16.MAX_STATE)
        surprise_sq_q = ALU_Q16.norm_sq(mean_err_q)

        # L3 Abstract Regime Update (Event-Driven Context)
        if surprise_sq_q > (ALU_Q16.ONE >> 2):
            self.slow_z_q += (new_states1_q[0] - self.slow_z_q) >> 4
            self.regime_hash = hashlib.sha256(self.slow_z_q.tobytes()).hexdigest()[:6]
            self.regime_stability = 0
        else:
            self.regime_stability += 1

        self.surprise_lp = (self.surprise_lp * 3 + surprise_sq_q) >> 2

        return new_states1_q, new_states2_q, self.surprise_lp, unc_sq_q


# =====================================================================
# 5. INTEGER CEM PLANNER (Continuous Actions)
# =====================================================================
@njit(cache=True, fastmath=True)
def _q16_cem_plan_jit(
    states1_array,
    states2_array,
    target_obs_q,
    explore_shift,
    n_cores,
    s_dim,
    s2_dim,
    a_dim,
    o_dim,
    cem_pop,
    horizon,
    cem_iters,
    W11_array,
    W12_array,
    W1a_array,
    C_array,
    ONE,
    MAX_STATE,
    SHIFT,
    seed_base,
):
    mu_q = np.zeros((horizon, a_dim), dtype=np.int64)
    std_q = np.ones((horizon, a_dim), dtype=np.int64) * ONE
    n_elite = max(1, cem_pop // 3)

    best_trajectory_q = np.zeros((horizon, a_dim), dtype=np.int64)
    best_futures_q = np.zeros((horizon, o_dim), dtype=np.int64)
    best_efe_q = np.int64(1) << np.int64(60)

    seed = seed_base

    for _ in range(cem_iters):
        noise_q = np.zeros((cem_pop, horizon, a_dim), dtype=np.int64)
        for i in range(cem_pop):
            for h in range(horizon):
                for d in range(a_dim):
                    v = np.int64(0)
                    for _k in range(3):
                        seed = _xorshift32(seed)
                        r = seed & np.int64(0xFFFF)
                        v += (r << (SHIFT - 15)) - (2 * ONE)
                    noise_q[i, h, d] = v // 3

        trajectories_q = np.zeros((cem_pop, horizon, a_dim), dtype=np.int64)
        for i in range(cem_pop):
            for h in range(horizon):
                for d in range(a_dim):
                    val = mu_q[h, d] + _alu_mul(noise_q[i, h, d], std_q[h, d])
                    if val > 2 * ONE:
                        val = 2 * ONE
                    elif val < -2 * ONE:
                        val = -2 * ONE
                    trajectories_q[i, h, d] = val

        scores_int = np.zeros(cem_pop, dtype=np.int64)

        for i in range(cem_pop):
            traj_q = trajectories_q[i]
            sim_s1 = np.zeros((n_cores, s_dim), dtype=np.int64)
            sim_s2 = np.zeros((n_cores, s2_dim), dtype=np.int64)
            for k in range(n_cores):
                for d in range(s_dim):
                    sim_s1[k, d] = states1_array[k, d]
                for d in range(s2_dim):
                    sim_s2[k, d] = states2_array[k, d]

            efe_q = np.int64(0)
            cur_futures_q = np.zeros((horizon, o_dim), dtype=np.int64)

            for h in range(horizon):
                a_q = traj_q[h]
                preds_q = np.zeros((n_cores, o_dim), dtype=np.int64)
                for k in range(n_cores):
                    s1_pred = (
                        _alu_mac(W11_array[k], sim_s1[k])
                        + _alu_mac(W12_array[k], sim_s2[k])
                        + _alu_mac(W1a_array[k], a_q)
                    )
                    for d in range(s_dim):
                        if s1_pred[d] > MAX_STATE:
                            s1_pred[d] = MAX_STATE
                        elif s1_pred[d] < -MAX_STATE:
                            s1_pred[d] = -MAX_STATE
                        sim_s1[k, d] = s1_pred[d]

                    preds_q[k] = _alu_mac(C_array[k], sim_s1[k])

                prag_sq_q = _alu_norm_sq(preds_q[0] - target_obs_q)

                mean_pred_q = np.zeros(o_dim, dtype=np.int64)
                for d in range(o_dim):
                    s = 0
                    for k in range(n_cores):
                        s += preds_q[k, d]
                    mean_pred_q[d] = (s + (n_cores >> 1)) // n_cores
                cur_futures_q[h] = mean_pred_q

                epist_sq_q = np.int64(0)
                for k in range(n_cores):
                    epist_sq_q += _alu_norm_sq(preds_q[k] - mean_pred_q)
                epist_sq_q = epist_sq_q // n_cores

                effort_sq_q = _alu_norm_sq(a_q)

                if explore_shift >= 0:
                    efe_q += (
                        prag_sq_q - (epist_sq_q << explore_shift) + (effort_sq_q >> 3)
                    )
                else:
                    efe_q += (
                        prag_sq_q
                        - (epist_sq_q >> abs(explore_shift))
                        + (effort_sq_q >> 3)
                    )

            scores_int[i] = efe_q
            if efe_q < best_efe_q:
                best_efe_q = efe_q
                for h in range(horizon):
                    for d in range(a_dim):
                        best_trajectory_q[h, d] = traj_q[h, d]
                    for d in range(o_dim):
                        best_futures_q[h, d] = cur_futures_q[h, d]

        idx = np.argsort(scores_int)

        for h in range(horizon):
            for d in range(a_dim):
                sum_mu = np.int64(0)
                for e in range(n_elite):
                    sum_mu += trajectories_q[idx[e], h, d]
                mu_q[h, d] = (sum_mu + (n_elite >> 1)) // n_elite

                sum_var = np.int64(0)
                for e in range(n_elite):
                    diff = trajectories_q[idx[e], h, d] - mu_q[h, d]
                    sum_var += _alu_mul(diff, diff)
                var_q = (sum_var + (n_elite >> 1)) // n_elite

                if var_q <= 0:
                    std_q[h, d] = 0
                else:
                    val = var_q << SHIFT
                    x = val
                    y = (x + 1) >> 1
                    while y < x:
                        x = y
                        y = (x + val // x) >> 1
                    std_q[h, d] = x

    return best_trajectory_q, best_futures_q


def q16_cem_plan(
    ensemble, states1_q, states2_q, target_obs_q, explore_shift, seed_base
):
    """Cross-Entropy Method Planner operating natively through LLVM JIT."""
    n_cores = config.n_cores
    s_dim = config.s_dim
    s2_dim = config.s2_dim
    a_dim = config.a_dim
    o_dim = config.o_dim
    horizon = config.horizon

    W11_array = np.zeros((n_cores, s_dim, s_dim), dtype=np.int64)
    W12_array = np.zeros((n_cores, s_dim, s2_dim), dtype=np.int64)
    W1a_array = np.zeros((n_cores, s_dim, a_dim), dtype=np.int64)
    C_array = np.zeros((n_cores, o_dim, s_dim), dtype=np.int64)
    states1_array = np.zeros((n_cores, s_dim), dtype=np.int64)
    states2_array = np.zeros((n_cores, s2_dim), dtype=np.int64)

    for k, core in enumerate(ensemble.cores):
        W11_array[k] = core.W11
        W12_array[k] = core.W12
        W1a_array[k] = core.W1a
        C_array[k] = core.C
        states1_array[k] = states1_q[k]
        states2_array[k] = states2_q[k]

    # Execute optimal trajectory derivation directly on native LLVM array processing
    best_traj_q, best_futures_q = _q16_cem_plan_jit(
        states1_array,
        states2_array,
        target_obs_q,
        np.int64(explore_shift),
        n_cores,
        s_dim,
        s2_dim,
        a_dim,
        o_dim,
        config.cem_pop,
        horizon,
        config.cem_iters,
        W11_array,
        W12_array,
        W1a_array,
        C_array,
        ALU_Q16.ONE,
        ALU_Q16.MAX_STATE,
        ALU_Q16.SHIFT,
        np.int64(seed_base),
    )

    return best_traj_q[0], best_futures_q


# =====================================================================
# 6. TRI-STATE SENTINEL GATE
# =====================================================================
class TriStateGateQ16:
    def __init__(self, shock_f=1.2, drift_f=0.8):
        self.shock_sq = ALU_Q16.to_q(shock_f**2)
        self.drift_sq = ALU_Q16.to_q(drift_f**2)
        self.ema_sq = 0
        self.causal_base_sq = 0

    def evaluate(self, surprise_sq_q, a_q):
        self.ema_sq = self.ema_sq - (self.ema_sq >> 3) + (surprise_sq_q >> 3)
        self.causal_base_sq = (
            self.causal_base_sq - (self.causal_base_sq >> 6) + (surprise_sq_q >> 6)
        )

        is_shock = surprise_sq_q > self.shock_sq
        is_drift = self.ema_sq > self.drift_sq
        effort_sq_q = ALU_Q16.norm_sq(a_q)
        is_causal = (surprise_sq_q > (self.causal_base_sq * 3)) and (
            effort_sq_q > (ALU_Q16.ONE >> 2)
        )

        return (is_shock or is_drift or is_causal), {
            "shock": is_shock,
            "drift": is_drift,
            "causal": is_causal,
        }

    def reset(self):
        self.ema_sq = 0
        self.causal_base_sq = 0


# =====================================================================
# 7. ENVIRONMENT & LOGGER
# =====================================================================
class Drone2D:
    def __init__(self):
        self.x, self.y, self.vx, self.vy = 0.0, 0.0, 0.0, 0.0
        self.mass, self.gravity, self.drag = 2.0, 0.5, 0.1

    def step(self, thrust_x, thrust_y):
        ax = (thrust_x / self.mass) - (self.drag * self.vx)
        ay = (thrust_y / self.mass) - self.gravity - (self.drag * self.vy)
        self.vx += ax * 0.5
        self.vy += ay * 0.5
        self.x += self.vx * 0.5
        self.y += self.vy * 0.5

        pixels = np.zeros(16)
        for i in range(4):
            for j in range(4):
                pixels[i * 4 + j] = np.exp(
                    -0.5 * ((self.x - j) ** 2 + (self.y - i) ** 2)
                )
        return pixels


class SDRProjector:
    def __init__(self, raw_dim=16, obs_dim=24, k_sparse=6):
        self.proj = np.random.randn(obs_dim, raw_dim) * (1.0 / np.sqrt(raw_dim))
        self.k = k_sparse

    def encode_q16(self, raw_sensor):
        x = self.proj @ raw_sensor
        idx = np.argsort(np.abs(x))[-self.k :]
        out = np.zeros_like(x)
        out[idx] = x[idx]
        return ALU_Q16.to_q(np.tanh(out))


class ApexLogger:
    def __init__(self):
        self.log = {"x": [], "y": [], "surp": [], "unc": [], "regime": [], "mode": []}
        self.fault_tick = None

    def record(self, env, surp_sq_q, unc_sq_q, regime_hash, recovery_mode):
        self.log["x"].append(env.x)
        self.log["y"].append(env.y)
        self.log["surp"].append(np.sqrt(ALU_Q16.to_f(surp_sq_q)))
        self.log["unc"].append(np.sqrt(ALU_Q16.to_f(unc_sq_q)))
        self.log["regime"].append(regime_hash)
        self.log["mode"].append(1.0 if recovery_mode else 0.0)

    def render(self, target_x, target_y):
        if not HAS_PLT:
            print("[LOGGER] Matplotlib not installed. Skipping visual dashboard.")
            return

        plt.style.use("dark_background")
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 1, height_ratios=[2, 1, 0.8])
        fig.suptitle(
            "Apex Sovereign V2: Autonomous Fault Recovery & Epistemic Adaptation",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Trajectory
        ax0 = fig.add_subplot(gs[0])
        ax0.plot(self.log["x"], self.log["y"], "c-", linewidth=2, label="Agent Path")
        ax0.scatter(target_x, target_y, c="lime", marker="*", s=300, label="Target")
        if self.fault_tick:
            ax0.scatter(
                self.log["x"][self.fault_tick],
                self.log["y"][self.fault_tick],
                c="red",
                s=250,
                marker="X",
                label="Motor Snapped",
            )
        ax0.set_title("2D Kinematic Trajectory")
        ax0.legend(loc="upper left")
        ax0.grid(True, alpha=0.3)

        # 2. Cognitive Dynamics
        ax1 = fig.add_subplot(gs[1])
        ax1.plot(
            self.log["surp"], "m-", label="Pragmatic Surprise (||e||)", linewidth=1.5
        )
        ax1.plot(self.log["unc"], "y--", label="Epistemic Uncertainty", linewidth=2)
        ax1.fill_between(
            range(len(self.log["mode"])),
            0,
            max(self.log["surp"]),
            where=np.array(self.log["mode"]) > 0,
            color="red",
            alpha=0.2,
            label="Gate Recovery Mode",
        )
        ax1.set_title("Cognitive Integers & Thermodynamic Free Energy")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # 3. L3 Regime Tracker
        ax2 = fig.add_subplot(gs[2])
        unique_hashes = list(dict.fromkeys(self.log["regime"]))
        hash_ids = [unique_hashes.index(h) for h in self.log["regime"]]
        ax2.step(range(len(hash_ids)), hash_ids, "g-", linewidth=2)
        ax2.set_title("L3 Abstract Regime Memory (SHA-256 Hashes)")
        ax2.set_yticks(range(len(unique_hashes)))
        ax2.set_yticklabels([f"0x{h}" for h in unique_hashes])
        ax2.set_xlabel("Hardware Ticks")

        plt.tight_layout()
        plt.show()


# =====================================================================
# 8. MASTER EXECUTION ORCHESTRATOR
# =====================================================================
class ApexSovereign:
    def __init__(self):
        self.env, self.sdr, self.logger = Drone2D(), SDRProjector(), ApexLogger()
        self.ensemble, self.gate = MetaEpistemicEnsemble(), TriStateGateQ16()
        self.states1_q = [
            np.zeros(config.s_dim, dtype=np.int64) for _ in range(config.n_cores)
        ]
        self.states2_q = [
            np.zeros(config.s2_dim, dtype=np.int64) for _ in range(config.n_cores)
        ]

        self.tgt_x, self.tgt_y = 2.0, 2.0
        tgt_pixels = np.zeros(16)
        for i in range(4):
            for j in range(4):
                tgt_pixels[i * 4 + j] = np.exp(
                    -0.5 * ((self.tgt_x - j) ** 2 + (self.tgt_y - i) ** 2)
                )
        self.target_obs_q = self.sdr.encode_q16(tgt_pixels)

    def run(self):
        print("=" * 80)
        print(
            "|| APEX SOVEREIGN V2: HARDENED 100% INTEGER ACTIVE INFERENCE ENGINE       ||"
        )
        print("=" * 80)

        recovery_mode = False
        recovery_timer = 0

        for t in range(250):
            # 🚨 Kinetic Fault Injection at T=120
            if t == 120:
                print(f"\n🚨 TICK {t}: FATAL MOTOR FAULT INJECTED (Drag increased 6x)")
                self.env.drag = 0.6
                self.logger.fault_tick = t

            # --- Meta-Control & CEM Planning ---
            if recovery_mode:
                active_lr = config.recov_lr_shift
                exp_shift = 2  # Max epistemic curiosity to re-learn broken physics
                a_q = ALU_Q16.to_q([0.0, 1.0])  # Override CEM Planner: Safe Hover
                recovery_timer -= 1
            else:
                active_lr = config.base_lr_shift
                # Autonomous L3 Meta-Control modulating exploration
                exp_shift = 1 if self.ensemble.regime_stability < 30 else -2
                a_q, _ = q16_cem_plan(
                    self.ensemble,
                    self.states1_q,
                    self.states2_q,
                    self.target_obs_q,
                    exp_shift,
                    np.int64(t ^ 42),
                )

            # --- Physical Environment Step ---
            raw_pixels = self.env.step(ALU_Q16.to_f(a_q)[0], ALU_Q16.to_f(a_q)[1])
            o_q = self.sdr.encode_q16(raw_pixels)

            # --- Cognitive Inference Step ---
            self.states1_q, self.states2_q, surp_lp_q, unc_sq_q = self.ensemble.step(
                self.states1_q, self.states2_q, a_q, o_q, active_lr
            )
            block, flags = self.gate.evaluate(surp_lp_q, a_q)

            # --- Sentinel Causal Fault Recovery ---
            if block and not recovery_mode and t > 50:
                print(f"  [T:{t}] 🛑 GATE TRIGGERED: Shock/Causal Anomaly Isolated!")
                print("          Initiating Epistemic Damage Adaptation...")
                recovery_mode = True
                recovery_timer = 20  # Force epistemic foraging for 20 ticks
                self.gate.reset()

            if recovery_mode and recovery_timer <= 0 and surp_lp_q < (ALU_Q16.ONE >> 1):
                print(
                    f"  [T:{t}] ✅ PHYSICS RE-ALIGNED. Lifting Gate Lock & Resuming Mission.\n"
                )
                recovery_mode = False

            self.logger.record(
                self.env, surp_lp_q, unc_sq_q, self.ensemble.regime_hash, recovery_mode
            )

            if t % 15 == 0 or t == 120 or (recovery_mode and recovery_timer == 10):
                m_str = (
                    "RECOVERING"
                    if recovery_mode
                    else ("FORAGING" if exp_shift > 0 else "TARGETING")
                )
                s_f = np.sqrt(ALU_Q16.to_f(surp_lp_q))
                print(
                    f"[{m_str:<10}] T:{t:03d} | Pos: ({self.env.x:>4.1f}, {self.env.y:>4.1f}) | Surp: {s_f:.3f} | Regime: [0x{self.ensemble.regime_hash}]"
                )

        self.logger.render(self.tgt_x, self.tgt_y)


if __name__ == "__main__":
    app = ApexSovereign()
    app.run()
