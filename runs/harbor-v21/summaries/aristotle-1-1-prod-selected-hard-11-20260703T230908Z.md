# Aristotle 1.1 Terminal-Bench 2.1 Selected Hard Summary

- Model: `trustedrouter/aristotle-1.1`
- Base URL: `https://api.trustedrouter.com/v1`
- Subset: Fast-Hard selected 3 + Slow-Hard selected 3 + Slowest-Hard selected 5
- Job: `aristotle-1-1-prod-selected-hard-11d-20260703T230908Z`
- Strict score: 1/11 = 9.09%
- Partial Macro: 3.000000/11 = 27.27%
- Partial Micro: 9/27 verifier tests = 33.33%
- Calls: 462
- Advisor calls: 24
- Cost: $85.639713
- Harbor runtime: 6h 32m 1s

Partial Macro is computed as the average of each task's verifier partial score (`passed_tests / total_tests`) from CTRF.

## Task Scores

| Task | Strict | Partial | Tests | Calls | Advisor calls | Cost | Trial time | Model time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model-extraction-relu-logits` | FAIL | 0.000000 | 0/1 | 8 | 1 | $0.575202 | 6m 31s | 5m 2s |
| `configure-git-webserver` | PASS | 1.000000 | 1/1 | 14 | 1 | $0.549131 | 4m 56s | 3m 28s |
| `cancel-async-tasks` | FAIL | 0.833333 | 5/6 | 7 | 0 | $0.206106 | 2m 36s | 1m 58s |
| `train-fasttext` | FAIL | 0.500000 | 1/2 | 60 | 0 | $5.942488 | 52m 20s | 7m 43s |
| `path-tracing-reverse` | FAIL | 0.666667 | 2/3 | 61 | 2 | $11.592272 | 26m 42s | 25m 27s |
| `extract-moves-from-video` | FAIL | 0.000000 | 0/2 | 60 | 0 | $6.408105 | 19m 5s | 6m 20s |
| `make-doom-for-mips` | FAIL | 0.000000 | 0/3 | 60 | 4 | $8.982250 | 28m 25s | 25m 10s |
| `protein-assembly` | FAIL | 0.000000 | 0/1 | 12 | 2 | $2.852112 | 19m 55s | 14m 34s |
| `gpt2-codegolf` | FAIL | 0.000000 | 0/1 | 60 | 6 | $23.412251 | 2h 23m 24s | 1h 45m 28s |
| `install-windows-3.11` | FAIL | 0.000000 | 0/4 | 60 | 4 | $11.342708 | 43m 51s | 14m 13s |
| `make-mips-interpreter` | FAIL | 0.000000 | 0/3 | 60 | 4 | $13.777088 | 44m 11s | 37m 38s |

## Strict Failures

- `model-extraction-relu-logits`: partial 0.000000 (0/1), calls 8, advisor calls 1; failed: `test_outputs.py::test_stolen_matrix_matches`.
- `cancel-async-tasks`: partial 0.833333 (5/6), calls 7, advisor calls 0; failed: `test_outputs.py::test_tasks_cancel_above_max_concurrent`.
- `train-fasttext`: partial 0.500000 (1/2), calls 60, advisor calls 0; failed: `test_outputs.py::test_accuracy`.
- `path-tracing-reverse`: partial 0.666667 (2/3), calls 61, advisor calls 2; failed: `test_outputs.py::test_image_similarity`.
- `extract-moves-from-video`: partial 0.000000 (0/2), calls 60, advisor calls 0; failed: `test_outputs.py::test_solution_file_exists`, `test_outputs.py::test_solution_content_similarity`.
- `make-doom-for-mips`: partial 0.000000 (0/3), calls 60, advisor calls 4; failed: `test_outputs.py::test_vm_execution`, `test_outputs.py::test_frame_bmp_exists`, `test_outputs.py::test_frame_bmp_similar_to_reference`.
- `protein-assembly`: partial 0.000000 (0/1), calls 12, advisor calls 2; failed: `test_outputs.py::test_gblock`.
- `gpt2-codegolf`: partial 0.000000 (0/1), calls 60, advisor calls 6; failed: `test_outputs.py::test_gpt2_implementation`.
- `install-windows-3.11`: partial 0.000000 (0/4), calls 60, advisor calls 4; failed: `test_outputs.py::test_network_status`, `test_outputs.py::test_qemu_running_with_correct_params`, `test_outputs.py::test_windows_311_core_files_verification`, `test_outputs.py::test_windows_keys_with_visual_feedback`.
- `make-mips-interpreter`: partial 0.000000 (0/3), calls 60, advisor calls 4; failed: `test_outputs.py::test_vm_execution`, `test_outputs.py::test_frame_bmp_exists`, `test_outputs.py::test_frame_bmp_similar_to_reference`.

## Artifacts

- Run: `runs/harbor-v21/aristotle-1-1-prod-selected-hard-11d-20260703T230908Z`
- Harbor result: `runs/harbor-v21/aristotle-1-1-prod-selected-hard-11d-20260703T230908Z/result.json`
- Replay manifest: `runs/harbor-v21/aristotle-1-1-prod-selected-hard-11d-20260703T230908Z/replay_manifest.json`
- JSON summary: `runs/harbor-v21/summaries/aristotle-1-1-prod-selected-hard-11-20260703T230908Z.json`
