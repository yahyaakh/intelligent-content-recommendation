import matplotlib.pyplot as plt
import numpy as np

# NDCG@10 on the 814 test users (Table classical-cold)
methods = ["Content-based", "Collaborative", "Hybrid", "SVD", "NCF (NeuMF)",
           "Item-CF", "Hybrid E", "User-kNN", "EASE"]
seeds10 = [0.319, 0.315, 0.350, 0.526, 0.598, 0.617, 0.631, 0.646, 0.651]
seeds3  = [0.230, 0.286, 0.359, 0.532, 0.547, 0.539, 0.597, 0.609, 0.614]
popularity = 0.560

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.0, 3.6))

y = np.arange(len(methods))
h = 0.38
ax.barh(y - h/2, seeds10, height=h, color="#4a4a4a", label="10 seeds")
ax.barh(y + h/2, seeds3,  height=h, color="#b0b0b0", label="3 seeds")
ax.axvline(popularity, color="black", linestyle="--", linewidth=1, label="Popularity")

ax.set_yticks(y)
ax.set_yticklabels(methods)
ax.set_xlabel("NDCG@10")
ax.set_xlim(0, 0.7)
ax.invert_yaxis()
# Legend above the plot, so it never covers a bar
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("thesis/thesis_latex/figures/classical_ndcg.pdf")
print("saved")