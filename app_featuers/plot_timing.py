import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1.  read data
# --------------------------------------------------
url = "graph_timings.csv"
df = pd.read_csv(url)
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

plt.style.use("seaborn-v0_8-whitegrid")

# --------------------------------------------------
# 2.  three quick plots  –– ALL BLUE
# --------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

# 2a. total latency sequence
ax[0].plot(df.index, df["total_time_ms"], marker="o", lw=1, ms=3, color="tab:blue")
ax[0].set_title("Total latency along image sequence")
ax[0].set_xlabel("image index")
ax[0].set_ylabel("total_time_ms")

# 2b. inference vs total scatter
ax[1].scatter(
    df["inference_time_ms"], df["total_time_ms"], alpha=0.7, s=25, color="tab:blue"
)
ax[1].set_xlabel("inference_time_ms")
ax[1].set_ylabel("total_time_ms")
ax[1].set_title("Inference vs total latency")
lims = [0, max(df["total_time_ms"].max(), df["inference_time_ms"].max())]
ax[1].plot(lims, lims, "k--", lw=1)  # keep guideline black

# 2c. histogram of total latency
ax[2].hist(df["total_time_ms"], bins=20, edgecolor="k", color="tab:blue", alpha=0.7)
ax[2].set_title("Distribution of total latency")
ax[2].set_xlabel("total_time_ms")
ax[2].set_ylabel("count")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 3.  INFERENCE-TIME DEEP-DIVE  –– ALL BLUE
# --------------------------------------------------
fig2, ax2 = plt.subplots(2, 2, figsize=(14, 8))
fig2.suptitle("Inference-time deep-dive", fontsize=16)

# 3a. inference sequence
ax2[0, 0].plot(
    df.index, df["inference_time_ms"], marker="o", lw=1, ms=3, color="tab:blue"
)
ax2[0, 0].set_title("Inference latency vs image index")
ax2[0, 0].set_xlabel("image index")
ax2[0, 0].set_ylabel("inference_time_ms")

# 3b. histogram
ax2[0, 1].hist(
    df["inference_time_ms"], bins=20, edgecolor="k", color="tab:blue", alpha=0.7
)
mean_val = df["inference_time_ms"].mean()
ax2[0, 1].axvline(mean_val, color="k", ls="--", lw=2, label=f"mean = {mean_val:.0f} ms")
ax2[0, 1].legend()
ax2[0, 1].set_title("Distribution of inference latency")
ax2[0, 1].set_xlabel("inference_time_ms")
ax2[0, 1].set_ylabel("count")

# 3c. boxplot
box = ax2[1, 0].boxplot(df["inference_time_ms"], vert=True, patch_artist=True)
box["boxes"][0].set_facecolor("tab:blue")  # colour the box
box["boxes"][0].set_alpha(0.7)
ax2[1, 0].set_ylabel("inference_time_ms")
ax2[1, 0].set_title("Boxplot (quartiles + outliers)")

# 3d. cumulative distribution
sorted_inf = df["inference_time_ms"].sort_values()
y = np.arange(1, len(sorted_inf) + 1) / len(sorted_inf) * 100
ax2[1, 1].plot(sorted_inf, y, lw=2, color="tab:blue")
ax2[1, 1].set_xlabel("inference_time_ms")
ax2[1, 1].set_ylabel("percentile")
ax2[1, 1].set_title("Cumulative distribution")
ax2[1, 1].grid(True, ls=":")

plt.tight_layout()
plt.show()
