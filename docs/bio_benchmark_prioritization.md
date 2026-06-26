# Bio Benchmark Prioritization

Assumptions: cost estimates are for one text model, one pass, constrained answer output, using a Sonnet-class hosted model at roughly `$3/M input + $15/M output`. Fusion or panel runs multiply cost by about `5-8x`. Time assumes API concurrency around `10`.

| Priority | Benchmark | Items | Run Cost | Run Time | Discrimination Signal | Notes |
|---:|---|---:|---:|---:|---|---|
| 1 | LAB-bench: LitQA2 + DbQA + SuppQA + ProtocolQA | 909 | `$1-3` | `10-25 min` | High | Best first target. Bio/lit/database/protocol QA, mostly multiple choice, easy exact scoring, enough items to separate models. |
| 2 | HLE Bio/Chem Gold | 149 | `$0.50-2` | `5-15 min` | Very high | Hard validated bio/chem questions. Smaller N, but strong quality. Some answers may need judge/normalization. |
| 3 | LAB-bench: SeqQA | 600 | `$2-5` | `10-25 min` | Medium-high | Sequence reasoning, long prompts. Good signal, but more token-heavy and may reward deterministic parsing as much as model knowledge. |
| 4 | ether0 | 325 | `$0.50-2` generation, more if judged | `5-15 min` | Medium-high | Chemistry/molecule generation. Cheap prompts, but proper SMILES/reward scoring adds harness complexity. |
| 5 | LAB-bench: FigQA + TableQA | 425 | `$1-4` text-only, higher vision | `15-40 min` | High if vision enabled | Valuable multimodal lab signal, but requires image-capable models and image input handling. Not first unless vision is a priority. |
| 6 | LAB-bench: CloningScenarios | 33 | `$0.50-2` | `5-10 min` | Medium | Very long plasmid prompts. Small N means noisy; useful as a targeted biology stress test, not a headline score. |
| 7 | BixBench | 205 | `$2-10+` before tools | `1-4 hr+` | Very high, once harnessed | Realistic research/data-analysis tasks with capsules. Best realism, worst setup/time. Save until cheap QA baselines are done. |
| 8 | SciCode full test | 288 substeps | `$$-$$$` | `hours` | High | Already run. Good science-coding signal, but expensive and slow compared with FutureHouse QA. |

## Recommended Run Packets

| Packet | Contents | Why |
|---|---|---|
| Smoke | 10 each from LitQA2, DbQA, SuppQA, ProtocolQA, HLE | Catches prompt/scoring bugs for cents. |
| Cheap leaderboard | 100-200 stratified LAB-bench text MC + all 149 HLE | Best cost/signal balance. |
| Expanded bio QA | Full LAB text MC + HLE + SeqQA | Strong model separation without capsules/tools. |
| Deep science agentic | BixBench or SciCode | Only after the cheap leaderboard identifies models worth spending on. |

## Recommendation

Implement and run the Cheap leaderboard first. It gives enough item count and difficulty diversity to discriminate models without paying for vision, molecule rewards, code execution, or BixBench capsules.
