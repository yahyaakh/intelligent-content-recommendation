# Plan for the next session (after 2026-10-25)

## Where we are

Everything up to now is written up in `thesis_documentation/13_experiment_llm_as_recommender.md`:

- **Idea 1, LLM reranks Hybrid E:** no LLM beats Hybrid E (12 models, 8 families). Test run: DeepSeek Flash −0.014, gemma4:e4b −0.031 (both significant). The best prompt is p2 at N = 10; adding the v4 themes and moods (p3) makes it significantly worse.
- **Idea 3, LLM reranks popularity:** **confirmed by a pre-registered test.** DeepSeek Flash, p2, N = 10, test users: +0.040 over popularity, p = 1.5 × 10⁻⁶. With thinking it was better still on validation (+0.044; +0.012 over no thinking, p = 0.025).
- **Idea 2, LLM recommends freely:** DeepSeek Flash with g2 at N = 10 equals popularity (0.574 vs 0.563, n.s.); gemma is worse.

## 1. Second pre-registered test: DeepSeek Flash with thinking

- [ ] Write down the test before running it: DeepSeek V4.1 Flash **with thinking**, idea 3 (popularity candidates), p2, N = 10. One test: LLM vs popularity on the 814 test users, Wilcoxon, α = 0.05. Report vs Hybrid E and vs the non-thinking test run descriptively.
- [ ] Cost about $5 off-peak (about 10,300 output tokens per user), about 6 hours (28 s per user). Run it off-peak (weekend, or before 14:00 / after 20:00 Budapest time; from 26 October winter time, 13:00 / 19:00) and in the background.
- [ ] `compare_models([("ollama_cloud", "deepseek-v4.1-flash", True)], data, split="test", version="p2", n_seeds=10, candidates="popularity")`

## 2. Verification of the idea-3 runs

- [ ] Extend `verify_llm_rerank.py` with the popularity candidates (`candidates` option, `_cpop` cache names) and run it on all idea-3 runs, validation and test.

## 3. Learning-to-rank (LightGBM / XGBoost), no API cost

- Same users, Hybrid E top-30 candidates; `LGBMRanker(objective="lambdarank")`, one group per user.
- Features: item-CF (mean, max), Hybrid E score and rank, popularity, TMDB votes; content and embedding similarity; LLM labels and 31b P2 profiles; **the LLM rankings from idea 1 (p2, N = 10) and idea 3** where answers exist for all users; genre, director and actor overlap with the seed profile.
- 5-fold CV on the 203 validation users, then one test run. Ablation: CF + popularity → + content → + LLM enrichment → + LLM rankings.
- Main question: do the LLM signals add information the ratings do not have?

## 4. Smaller follow-ups (only if time allows)

- [ ] Beyond-accuracy metrics (doc 12) for ideas 1 and 3: coverage, novelty, long-tail share.
- [ ] Two or three shuffles per user, averaged, to reduce position bias.
- [ ] Idea 2 with thinking (DeepSeek Flash) on validation.

## Housekeeping

- [ ] Remove the dead `"github"` provider from `PROVIDERS` (GitHub Models was retired in July 2026).
- [ ] Commit the scripts and docs 11–13 (nothing committed yet). Do not commit `.env` (it is in `.gitignore`).
