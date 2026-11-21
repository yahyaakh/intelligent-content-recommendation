import matplotlib.pyplot as plt

# DeepSeek V4.1 Flash, prompt p2, 10 seeds, 203 validation users (NDCG@10)
# Source: Chapter 6, tab:llm-models and the free-generation paragraph
roles = ["Rerank Hybrid E top 30", "Rerank popularity top 30", "Free generation"]
ndcg = [0.621, 0.594, 0.574]
hybrid_e = 0.629
popularity = 0.563

plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, ax = plt.subplots(figsize=(6.0, 2.4))

y = list(range(len(roles)))

# Each role is a line from popularity to its score: the length is what the LLM adds
for v, yi in zip(ndcg, y):
    ax.plot([popularity, v], [yi, yi], color="#4a4a4a", linewidth=2.5, zorder=2)
    ax.scatter(v, yi, s=60, color="#4a4a4a", zorder=3)
    # Value above the dot, so no reference line can cross it
    ax.text(v, yi - 0.18, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

# Reference lines, labelled directly at the top instead of in a legend
for x, style, name in [(popularity, "--", "Popularity"), (hybrid_e, "-", "Hybrid E")]:
    ax.axvline(x, color="black", linestyle=style, linewidth=1, zorder=1)
    # Label just above the plot area, where the line does not reach
    ax.text(x, 1.02, f"{name} ({x:.3f})", ha="center", va="bottom", fontsize=9,
            transform=ax.get_xaxis_transform())

ax.set_yticks(y)
ax.set_yticklabels(roles)
ax.set_ylim(len(roles) - 0.5, -0.7)   # first role on top, room for the value labels
ax.set_xlim(0.55, 0.645)
ax.set_xlabel("NDCG@10")
ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("thesis/thesis_latex/figures/llm_roles_ndcg.pdf", bbox_inches="tight")
print("saved")
