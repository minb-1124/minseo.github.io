"""
Completion-Sense Benchmark

각 환경(tiny/small/medium)에서 실제 NASim optimal path를 실행
→ obs 파싱 + 텍스트 변환 → ST embedding → reward 계산
→ FTA, Jump Detection, 단조증가 검증
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import nasim
import numpy as np

from config import GOAL_TEXT
from core.obs_to_text import obs_to_text
from core.semantic_encoder import (
    semantic_reward_with_baseline,
    set_initial_state,
)


OPTIMAL_PATHS = {
    "tiny": {
        "actions": [4, 2, 16, 17, 10, 11],
        "labels": [
            "Initial",
            "Exploit SSH (1,0)",
            "Subnet Scan",
            "Exploit SSH (3,0)",
            "Root (3,0)",
            "Exploit SSH (2,0)",
            "Root (2,0)",
        ],
    },
    "small": {
        "actions": [6, 2, 13, 16, 33, 29, 67, 70],
        "labels": [
            "Initial",
            "Exploit HTTP (1,0)",
            "Subnet Scan (1,0)",
            "Exploit SSH (2,0)",
            "Root (2,0)",
            "Exploit HTTP (3,1)",
            "Subnet Scan (3,1)",
            "Exploit SSH (4,0)",
            "Root (4,0)",
        ],
    },
    "medium": {
        "actions": [6, 2, 20, 23, 42, 38, 148, 151],
        "labels": [
            "Initial",
            "Exploit HTTP (1,0)",
            "Subnet Scan (1,0)",
            "Exploit SMTP (2,0)",
            "Root (2,0)",
            "Exploit HTTP (3,1)",
            "Subnet Scan (3,1)",
            "Exploit SSH (5,0)",
            "Root (5,0)",
        ],
    },
}


def get_stage_obs(scenario_id: str, max_retries: int = 5):
    """
    시나리오별 optimal path를 실행하여 각 단계의 obs 수집.
    확률적 exploit 실패 시 재시도.
    """
    path_info = OPTIMAL_PATHS[scenario_id]
    actions = path_info["actions"]
    labels = path_info["labels"]

    stages = []
    n = 0
    for _attempt in range(max_retries):
        env = nasim.make_benchmark(scenario_id, flat_obs=True, fully_obs=True)
        obs, _ = env.reset()
        n = env.action_space.n

        stages = [("Initial", obs.copy())]

        for i, a in enumerate(actions):
            obs, _r, done, _trunc, _info = env.step(int(a))
            stage_label = labels[i + 1]
            stages.append((stage_label, obs.copy()))

            if done and i < len(actions) - 1:
                for j in range(i + 2, len(labels)):
                    stages.append((labels[j], obs.copy()))
                break

        env.close()

        if len(stages) == len(labels):
            return stages, n

    print(f"  [WARNING] {scenario_id}: optimal path failed after {max_retries} retries")
    return stages, n


def run_completion_benchmark(
    scenario_id: str,
    goal_text: str = GOAL_TEXT,
    n_runs: int = 6,
):
    path_info = OPTIMAL_PATHS[scenario_id]
    labels = path_info["labels"]
    n_stages = len(labels)

    print(f"\n{'=' * 60}")
    print("  Completion-Sense Benchmark")
    print(f"  Scenario : {scenario_id} ({n_stages - 1} steps)")
    print(f"  Goal  : '{goal_text}'")
    print(f"  Runs  : {n_runs}")
    print(f"{'=' * 60}")

    all_rewards = []
    all_sim_current = []
    all_texts = []

    for run in range(n_runs):
        print(f"\n  [Run {run + 1}] obs 수집 및 텍스트 변환 중...")
        stages, n_actions = get_stage_obs(scenario_id)

        run_rewards, run_cur, run_texts = [], [], []

        for i, (label, obs) in enumerate(stages):
            text = obs_to_text(
                obs,
                action_space_n=n_actions,
                prev_action_name=label,
                step=i,
                scenario_id=scenario_id,
            )

            if i == 0:
                set_initial_state(text)

            r = semantic_reward_with_baseline(text, goal_text)
            run_rewards.append(r["reward"])
            run_cur.append(r["potential"])
            run_texts.append(text)

            print(
                f"    stage {i} [{label:22s}] R={r['reward']:+.4f} "
                f"Phi={r['potential']:+.4f} sig={r['sigmoid_val']:.3f} "
                f"dPhi={r['delta_phi']:.4f} | {text[:45]}"
            )

        all_rewards.append(run_rewards)
        all_sim_current.append(run_cur)
        all_texts.append(run_texts)

    avg_rewards = np.mean(all_rewards, axis=0)
    avg_cur = np.mean(all_sim_current, axis=0)

    r_min = avg_rewards.min()
    r_max = avg_rewards.max()
    if r_max - r_min > 1e-8:
        avg_rewards_norm = (avg_rewards - r_min) / (r_max - r_min)
    else:
        avg_rewards_norm = np.zeros_like(avg_rewards)

    pcts = [int(i / (n_stages - 1) * 100) for i in range(n_stages)]

    print(f"\n{'단계':>2}  {'완료율':>5}  {'R(raw)':>8}  {'R(norm)':>8}  {'막대':20}  설명")
    print("-" * 80)
    for i, label in enumerate(labels):
        rn = avg_rewards_norm[i]
        bar = "#" * int(rn * 20)
        print(
            f"  {i:>2}  {pcts[i]:>4}%  "
            f"{avg_rewards[i]:+.4f}  "
            f"{rn:.4f}  {bar:20s}  {label}"
        )

    print("\n--- Forward Transition Accuracy ---")
    forward, total = 0, 0
    for run_rewards in all_rewards:
        for i in range(1, len(run_rewards)):
            total += 1
            if run_rewards[i] >= run_rewards[i - 1]:
                forward += 1
    fta = f"{forward}/{total}"
    print(f"  FTA : {fta}")

    print("\n--- Jump Detection ---")
    threshold = 0.05
    jump_detected = 0
    for idx, run_rewards in enumerate(all_rewards):
        delta = run_rewards[-1] - run_rewards[-2]
        detected = delta >= threshold
        if detected:
            jump_detected += 1
        marker = "detected" if detected else "missed"
        print(f"  Run {idx + 1}: delta={delta:+.4f}  {marker}")
    print(f"\n  Jump Detection : {jump_detected}/{n_runs}")

    print("\n--- Monotonicity ---")
    monotone_count = 0
    for idx, run_rewards in enumerate(all_rewards):
        is_mono = all(
            run_rewards[i] <= run_rewards[i + 1]
            for i in range(len(run_rewards) - 1)
        )
        if is_mono:
            monotone_count += 1
        marker = "monotone" if is_mono else "non-monotone"
        print(f"  Run {idx + 1}: {marker}")
    print(f"\n  Monotonicity : {monotone_count}/{n_runs}")

    return {
        "scenario": scenario_id,
        "n_stages": n_stages,
        "labels": labels,
        "avg_rewards": avg_rewards_norm.tolist(),
        "avg_rewards_raw": avg_rewards.tolist(),
        "avg_potential_raw": avg_cur.tolist(),
        "fta": fta,
        "jump": f"{jump_detected}/{n_runs}",
        "monotone": f"{monotone_count}/{n_runs}",
        "all_texts": all_texts,
        "all_rewards": [r for r in all_rewards],
    }


if __name__ == "__main__":
    import json
    from datetime import datetime

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("./logs", exist_ok=True)

    all_results = []
    scenarios = ["tiny", "small", "medium"]
    colors = {"tiny": "#4C72B0", "small": "#DD8452", "medium": "#55A868"}

    for scenario in scenarios:
        result = run_completion_benchmark(scenario)
        all_results.append(result)

    log_path = f"./logs/eval_{timestamp}.json"
    log_data = []
    for result in all_results:
        entry = {
            "scenario": result["scenario"],
            "n_stages": result["n_stages"],
            "labels": result["labels"],
            "avg_rewards": result["avg_rewards"],
            "avg_rewards_raw": result["avg_rewards_raw"],
            "avg_potential_raw": result["avg_potential_raw"],
            "fta": result["fta"],
            "jump": result["jump"],
            "monotone": result["monotone"],
            "runs": [],
        }
        for texts, rewards in zip(result["all_texts"], result["all_rewards"]):
            run_entry = []
            for stage_idx, (text, reward) in enumerate(zip(texts, rewards)):
                run_entry.append(
                    {
                        "stage": stage_idx,
                        "label": result["labels"][stage_idx],
                        "reward": round(float(reward), 6),
                        "text": text,
                    }
                )
            entry["runs"].append(run_entry)
        log_data.append(entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    print(f"\n  Log saved -> {log_path}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 300,
        }
    )
    markers = {"tiny": "o", "small": "s", "medium": "D"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, result, scenario in zip(axes, all_results, scenarios):
        rewards = result["avg_rewards"]
        n = len(rewards)
        pcts = [int(i / (n - 1) * 100) for i in range(n)]

        ax.plot(
            range(n),
            rewards,
            marker=markers[scenario],
            color=colors[scenario],
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        ax.set_title(f"{scenario.capitalize()} ({n - 1} steps)")
        ax.set_xlabel("Completion (%)")
        if ax == axes[0]:
            ax.set_ylabel("Normalized Semantic Reward")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"{p}" for p in pcts])
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.5)
    graph1 = f"./logs/completion_benchmark_{timestamp}.png"
    plt.savefig(graph1, dpi=300, bbox_inches="tight")
    print(f"  Graph saved -> {graph1}")

    fig2, ax2 = plt.subplots(figsize=(7, 4.5))

    for result, scenario in zip(all_results, scenarios):
        rewards = result["avg_rewards"]
        n = len(rewards)
        x_norm = [i / (n - 1) * 100 for i in range(n)]
        ax2.plot(
            x_norm,
            rewards,
            marker=markers[scenario],
            color=colors[scenario],
            linewidth=2,
            markersize=6,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=f"{scenario.capitalize()} ({n - 1} steps)",
        )

    ax2.set_xlabel("Normalized Completion (%)")
    ax2.set_ylabel("Normalized Semantic Reward")
    ax2.set_ylim(0, 1.05)
    ax2.legend(loc="lower right", framealpha=0.9, edgecolor="gray")
    ax2.grid(True, alpha=0.2, linewidth=0.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    graph2 = f"./logs/completion_comparison_{timestamp}.png"
    plt.savefig(graph2, dpi=300, bbox_inches="tight")
    print(f"  Graph saved -> {graph2}")

    fig3, ax3 = plt.subplots(figsize=(5, 4))

    scenario_labels = [s.capitalize() for s in scenarios]
    fta_fractions = []
    for result in all_results:
        numer, denom = map(int, result["fta"].split("/"))
        fta_fractions.append((numer, denom))
    fta_ratios = [numer / denom for numer, denom in fta_fractions]

    bars = ax3.bar(
        scenario_labels,
        fta_ratios,
        color=[colors[s] for s in scenarios],
        width=0.45,
        edgecolor="white",
        linewidth=1.5,
    )

    for bar, (numer, denom) in zip(bars, fta_fractions):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{numer}/{denom}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax3.set_xlabel("Scenario")
    ax3.set_ylabel("FTA")
    ax3.set_ylim(0, 1.05)
    ax3.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.4,
        label="Random (0.5)",
    )
    ax3.legend(loc="upper right", framealpha=0.9, edgecolor="gray")
    ax3.grid(True, axis="y", alpha=0.2, linewidth=0.5)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()
    graph3 = f"./logs/fta_comparison_{timestamp}.png"
    plt.savefig(graph3, dpi=300, bbox_inches="tight")
    print(f"  Graph saved -> {graph3}")

    print("\n" + "=" * 60)
    print("  Overall Results")
    print("=" * 60)
    print(f"  {'Scenario':>8}  {'Steps':>5}  {'FTA':>8}  {'Jump':>8}  {'Monotone':>10}")
    print("-" * 55)
    for result in all_results:
        print(
            f"  {result['scenario']:>8}  "
            f"{result['n_stages'] - 1:>5}  "
            f"{result['fta']:>8}  "
            f"{result['jump']:>8}  "
            f"{result['monotone']:>10}"
        )
