# Critical Analysis

**Date:** 2026-05-04
**Commit:** 7c25ae0 (working tree dirty: codebook.py, kv_cache.py, quantizer.py)
**Reviewer:** Claude Code (automated)
**Note:** Project is Rust-template-targeted but is a Python/mlx codebase. Validation commands were adapted: `cargo check`→smoke import; `cargo test`→`pytest`; `cargo clippy`→`ruff check`; `cargo doc`→docstring coverage script. There is no `Cargo.toml`, `docs/LLM_RULES.md`, or `docs/ARCHITECTURE.md`; equivalent context was taken from `README.md` and `structure.md`.

---

## Composite Score: 6.0 / 10

| Dimension | Score | Severity |
|-----------|-------|----------|
| 1. Safety & Correctness | 5/10 | High |
| 2. Error Handling | 6/10 | Medium |
| 3. API Design | 7/10 | Medium |
| 4. Concurrency | 8/10 | Low |
| 5. Testing | 4/10 | High |
| 6. Performance | 5/10 | High |
| 7. Documentation | 7/10 | Medium |
| 8. CI/CD & Release | 2/10 | Critical |
| 9. Dependency Hygiene | 7/10 | Medium |
| 10. Conventions | 5/10 | High |

Severity column: **Critical** = score 1-3, **High** = 4-5, **Medium** = 6-7, **Low** = 8-9, **None** = 10.

---

## Top 3 Blockers

1. **No CI pipeline whatsoever** — there is no `.github/`, no test workflow, no lint gate, no Apple-Silicon runner; every regression has to be caught by hand on a local M-series box. (`<repo root>` — missing `.github/workflows/*.yml`)
2. **Test suite is shape-only** — 10 unit tests across two files exercise dataclass shapes and a tolerance-loose unbiasedness check, but there is **no end-to-end test** that drives `patch_model` + `generate` and asserts numerical agreement vs. baseline; the entire decode-mode bug we just fixed (`_is_prefill` never flipping; missing RoPE; quadratic re-ingestion; wrong softmax scale) was invisible to the suite. (`tests/test_quantizer.py`, `tests/test_kv_cache.py`)
3. **`pack_bits` correctness is fragile and pessimistic** — the float32-matmul scatter in `codebook.py:299-319` only stays exact because contributions are bounded; for 5+ bit packing the assumption breaks silently, and there is no test asserting this. The unpack path also recomputes `lo_byte`/`hi_byte` plans that materialise as MLX arrays at module-import time per (bits, D) but are never freed — fine in practice, undocumented.

---

## Dimension Findings

### 1. Safety & Correctness — 5/10

The library has fixed several severe correctness issues recently (decode mode never engaged, quadratic ingestion, wrong softmax scale, missing RoPE in decode), but the current state still has structural risks. There is no integration test that would catch regressions of these same bugs. `attention()` silently coerces back to float16 at the end without checking for NaN/Inf, and `_decode_forward` swallows offset-write failures (`patch.py:174-178`).

The boundary-counting fix in `quantize_with_codebook` (`codebook.py:155`) corrected an off-by-one that would have shifted every quantised value by one bin — a nasty class of bug that should have a regression test pinned to a known input/output pair.

`_pack_plan` and `_unpack_plan` are computed eagerly and cached in module-level dicts; they're not thread-safe. mlx-lm is not currently multithreaded so this is not a runtime hazard, but it's a future-correctness footgun.

`assert k_buf is not None and v_buf is not None` (`kv_cache.py:132`) will silently disappear under `python -O`. That assertion is the only thing keeping `_flush` from a `None.shape` AttributeError if invariants drift; it should be a real check (or eliminated by typing the buffers properly).

**Issues:**
- `kv_cache.py:132` — runtime invariant guarded only by `assert`; vanishes under `-O` [Medium]
- `patch.py:174-178` — bare `except (AttributeError, TypeError): pass` swallows cache-offset write failures silently; if `cache.offset` becomes a property without a setter on a future mlx-lm version, this drops position tracking with no warning [Medium]
- No NaN/Inf guard in the decode hot path; one bad codebook entry (e.g., from a malformed codebook JSON edited by hand) silently propagates through 60 layers [Medium]
- `pack_bits` correctness assumes (a) `contrib` < 2^16 per element and (b) at most two coordinates per destination byte; both true for `bits ∈ {2,3,4}`, but no `assert` and no test pins this. A future contributor adding `bits=5` would silently corrupt outputs (`codebook.py:299-319`) [High]

---

### 2. Error Handling — 6/10

Errors are mostly raised as `NotImplementedError` or `ValueError` at sensible places (e.g., `patch.py:115` when a layer lacks `q_proj`). But several silent-fallback paths look more dangerous than helpful:

- `patch.py:101` returns `(out, kv_pair, offset_out)` from prefill *unchanged* if `kv_pair is None and cache is None` — TQ buffer simply never gets populated and decode will use an empty cache; no warning.
- `codebook.py` reads JSON codebooks with no schema validation. A truncated or hand-edited file would either crash with `KeyError` or silently load wrong-length boundary arrays.
- `cli/chat.py:30` regex matches `<|channel>thought\n.*?<channel|>` with no fallback if the model emits a different end token; the closer matches by chance on harmless-looking text could erase real assistant content from history.

**Issues:**
- `codebook.py:114-126` — JSON codebook load has no schema/length validation [Medium]
- `patch.py:91-99` — silent no-op when neither `kv_pair` nor `cache.keys` is present; produces an empty TQ cache and incorrect generation with no error [High]
- `cli/chat.py:30` — thinking-tag regex assumes specific delimiters; mismatched closer truncates response silently [Low]
- `kv_cache.py:140-145` — `_flush` does not validate that `n` ≤ `buffer_len`; trusting caller is fine, but if the buffer is empty (`buffer_len == 0`) and `n > 0` this raises an opaque slicing error [Low]

---

### 3. API Design — 7/10

The public surface is concise and well-shaped: `patch_model`, `set_decode_mode`, `set_prefill_mode`, plus the underlying `TurboQuantMSE`/`TurboQuantProd` and dataclasses for storage. Names are consistent with the paper. The wrapper preserves the gemma4_text `(out, kv_pair, offset)` calling convention so it slots into mlx-lm's generate loop transparently.

Weaknesses:
- `patch_model` mutates the model in-place and returns wrapper handles, but `set_decode_mode` / `set_prefill_mode` need those handles, and there is no API to retrieve them later from a patched model — callers either capture the return value at patch time or lose the ability to flip modes. The auto-flip in `patch.py:58-59` papers over this for the common case but does not eliminate the wart.
- `TurboQuantLayerCache.scale` is `Optional[float]` and silently defaults to `1/sqrt(D)`; this is a footgun because gemma4 uses `1.0` and forgetting to set it produces gibberish (this exact bug was a P0 last round). It should be a required parameter — no sensible default exists.
- `CompressedKey.qjl_signs` and `residual_norms` are `Optional` and *unused* on the decode hot path now (after the recent refactor that uses `mx.fast.scaled_dot_product_attention` with reconstructed keys). They still consume memory at compress time. Either remove the QJL path entirely or expose a flag to skip computing it.

**Issues:**
- `kv_cache.py:59` — `scale` defaulting to `1/sqrt(D)` is a silent footgun; gemma4 needs 1.0 [Medium]
- `quantizer.py:CompressedKey.qjl_signs` — unused on the decode hot path, still computed and stored every flush [Medium]
- `patch.py:204-258` — no API to fetch wrappers from a patched model; callers must capture the return value [Low]
- Public re-exports in `__init__.py` lack `__all__` (only the list assignment, no docstrings on the module) [Low]

---

### 4. Concurrency — 8/10

mlx-lm runs single-threaded with the GPU driver doing its own scheduling, so there is little real concurrency surface. The two module-level dict caches (`_CACHE` in `codebook.py:90`, `_PACK_PLAN`/`_UNPACK_PLAN` in `codebook.py:183-184`) are populated lazily and not protected; in a hypothetical multi-process or multi-threaded import they could race. Codebook generation is also non-deterministic across `lloyd_max(rng_seed=0)` invocations because `rng = np.random.default_rng(rng_seed)` is computed but never used (`codebook.py:49` — flagged by ruff as `F841`). That means seed parameter is decorative; codebooks generated locally vs. in CI would still match because the algorithm is deterministic without any RNG, but the API lies about being seedable.

**Issues:**
- `codebook.py:49` — `rng_seed` parameter is plumbed through but the RNG is never used; misleading API [Low]
- Module-level caches lack thread-safety hints; documented assumption that imports happen single-threaded [Low]

---

### 5. Testing — 4/10

10 tests across two files. Each test runs in <1s; that is good, but coverage is shallow.

What's missing:
- **No end-to-end model test.** Nothing exercises `patch_model` against a real (even tiny) gemma-style model and asserts that decode produces logits within tolerance of baseline. Every recent regression (RoPE, scale, quadratic ingestion, decode-mode flip) was unit-test invisible.
- **No round-trip test for `pack_bits`/`unpack_bits` across `(bits, D)` matrix.** The current `test_quantizer.py::test_compression_ratio` only checks the size of the packed output; it does not verify that unpack(pack(x)) == x for arbitrary D. (We did this manually in the conversation but did not commit it.)
- **No GQA/MQA test.** `TurboQuantLayerCache` has a non-trivial GQA reshape path (`kv_cache.py:_materialise_kv` reshapes by `(K, D)`) and zero tests cover `n_kv_heads > 1`. The KV-cache test fixtures use `n_kv_heads=1`.
- **No flush-correctness test.** `test_flush_triggers_on_overflow` checks that the *list* grew; nothing checks that the right tokens were compressed and the buffer slice afterwards is correct.
- **No test for `_strip_thinking` regex** in `cli/chat.py`.
- **`pytest` is imported in two test files but unused** (ruff `F401`); `ck` and `T` are bound but never used (`F841`). Linter-warning noise indicates the suite is not under any quality gate.
- **No `pytest-benchmark` use** even though it is a declared dev dep — there is no perf regression guard, which is exactly the wrong tradeoff for a project whose entire reason for existing is performance.

**Issues:**
- `tests/` — no end-to-end integration test against a patched model [Critical]
- `tests/test_quantizer.py` — no `pack_bits`/`unpack_bits` round-trip test [High]
- `tests/test_kv_cache.py` — all fixtures use `n_kv_heads=1`; GQA path untested [High]
- `tests/` — no benchmark covered by `pytest-benchmark` despite it being a dev dep [Medium]
- `tests/test_quantizer.py:14`, `tests/test_kv_cache.py:12` — `pytest` imported but unused [Low]

---

### 6. Performance — 5/10

Recent fixes corrected three catastrophic bugs (numpy `.tolist()` round-trip in pack/unpack; quadratic re-ingestion; per-token `mx.concatenate`). But the current state still has known structural inefficiencies the user has explicitly flagged:

- **Compressed-V is decompressed every decode step.** `_materialise_kv` (`kv_cache.py:190`) calls `value_quantizer.decompress(cv)` for every compressed batch on every step. This is the dominant cost at long context. Without a fused score-on-packed-bits Metal kernel — the paper's CUDA approach — TQ cannot beat baseline tokens/sec on Apple Silicon. `README.md:138` acknowledges this.
- **`_scatter_or_row` uses a `(D, packed_d)` one-hot bridge as a float32 matmul.** Correct; not particularly fast. For D=160, packed_d=60 (3-bit), this is a 9.6KB float32 matmul per pack/unpack call. Negligible at flush time, but unpack runs in the decode hot path so it adds up — though only 60 layers × 1 unpack per compressed batch, not per token.
- **`pack_bits` allocates a fresh `mx.zeros((n, packed_d))` each call** rather than fusing into the matmul output (`codebook.py:289`). Minor.
- **No `pytest-benchmark` guard.** Performance regressions are invisible.
- **`patch.py:96-97` does `keys[0, :, -T_new:, :].transpose(1, 0, 2).reshape(...)` per prefill call.** Fine once-per-layer for a single prompt, but if prefill happens in chunks the slicing creates non-contiguous views that may or may not stay zero-copy under MLX.

**Issues:**
- `kv_cache.py:190-217` — V decompressed every decode step; primary speed bottleneck without a Metal kernel [High]
- No benchmark suite to detect regressions [High]
- `codebook.py:317` — `(contrib @ onehot)` materialises a fresh float32 array each call; could be pre-computed and reused per `(bits, D)` [Low]

---

### 7. Documentation — 7/10

Strengths:
- Every module begins with a `# file:` / `# description:` / `# reference:` header. Good practice.
- README is concrete: paper citation, model link, exact `head_dim=160`, sampling defaults.
- `structure.md` is a clear file-by-file map.
- 22 of 23 public functions/classes have a docstring (only `codebook.cli_main` lacks one).

Weaknesses:
- README's `Limitations` section claims "Bit-packing and codebook lookups go through numpy on CPU" — **stale**, since the recent rewrite moved them to pure MLX (`codebook.py:264-342`). Same paragraph claims "Prefill uses standard mlx-lm attention; TurboQuant activates at decode time" — that statement is correct now but was incorrect for several iterations.
- README's "## Pre-generate codebooks" section says `head_dim=160` but the model the user actually loads in benchmarks (`mlx-community/gemma-4-31b-it-4bit`) has **`head_dim=256` for sliding and `512` for global** per the comment in `kv_cache.py:333`. Inconsistent.
- No CHANGELOG.
- No QUICKSTART beyond the README.
- `proof.py` references a model named `gemma-4-31b-it` that does not appear to be a real public release. The label `"[llama-cli ref] gemma-4-26B-A4B Q8_0"` in the proof script also references a non-existent model. Likely a placeholder vestige.
- No examples directory; the inline README example does not run end-to-end (no error path, no peak-memory tracking).

**Issues:**
- `README.md:138-141` — "Limitations" section is stale (numpy CPU path no longer present) [Medium]
- `README.md:43-44` — head_dim=160 advice contradicts kv_cache.py:333 (256/512 for the actual model) [Medium]
- `proof.py:18, 95` — dead/aspirational references to non-existent gemma-4-26B-A4B [Low]
- `codebook.py:cli_main` — undocumented [Low]
- No CHANGELOG / QUICKSTART [Low]

---

### 8. CI/CD & Release — 2/10

There is **no `.github/` directory at all.** No CI workflow, no scheduled lint, no automated test on PR, no Apple-Silicon runner test, no version-bump release automation, no semver enforcement. Every change is "did the local run pass." For a research/learning project this is acceptable; for the implied production-readiness scoring of this review, it's the dominant blocker.

`pyproject.toml` declares `tq-chat`, `tq-generate`, `tq-codebook` as console scripts; none of them have a `--help` smoke test in CI. The package is `version = "0.2.0"` with no tag, no GitHub release, no PyPI publication.

**Issues:**
- No CI workflow [Critical]
- No release automation, no version tags [Critical]
- No lint or type-check gate [High]

---

### 9. Dependency Hygiene — 7/10

`pyproject.toml` declares only three runtime deps: `mlx>=0.31.2`, `mlx-lm>=0.31.3`, `numpy>=2.4`. Lower bounds are recent and reasonable. The `[server]` extra adds `fastapi` + `uvicorn` but **no server module exists** in `src/turboquant_mlx/` — that extra is a dead promise.

`structure.md:213-218` claims `mlx >= 0.22` and `numpy >= 1.26` — disagrees with `pyproject.toml`. Documentation drift.

There is no `pip-audit` / `safety` invocation in CI (because there is no CI). `uv.lock` is committed (good) and 212KB, fine. No `Cargo.toml`-style license field per dependency, but Python doesn't really do that.

**Issues:**
- `pyproject.toml:20-23` — `[server]` extra is declared but no server module exists [Medium]
- `structure.md:213-218` vs. `pyproject.toml:9-13` — version requirements disagree [Low]
- No security advisory scan [Medium]

---

### 10. Conventions — 5/10

ruff finds **10 errors** without any custom rules enabled. Specifically:
- `F401`: 5 unused imports (`math` in two files, `pytest` in two test files, `Optional`, `field`)
- `F841`: 4 unused locals (`rng`, `K`, `T`, `ck`)

These are textbook "no project lint gate" smells. None of them are wrong-result bugs — but `rng_seed` being plumbed through `lloyd_max(rng_seed=0)` while `rng` is unused (`codebook.py:49`) is a signal lie that violates the "leave no dead code" rule explicitly stated in this review's prompt.

The `cli/chat.py` module mixes `_strip_thinking` regex assumptions with the chat REPL; the regex is hard to test because it's coupled to the REPL.

The recent linter-applied edit to `codebook.py:155` made a real correctness fix (`>=` upper-boundary count) without a corresponding test to prevent regression.

`# ty:ignore[invalid-assignment]` markers appear in scripts and CLI files referencing the **`ty` type checker**. There is no `ty` configured in `pyproject.toml` and no CI invocation. The markers are aspirational — they suggest a type-check pass was intended but never wired up.

**Issues:**
- `codebook.py:49` — unused `rng` [Low]
- `kv_cache.py:15` — unused `Optional` import [Low]
- `kv_cache.py:177` — unused local `K` in `attention()` [Low]
- `quantizer.py:10` — unused `field` import [Low]
- `quantizer.py:182` — unused local `T` [Low]
- `rotation.py:9` — unused `math` import [Low]
- `tests/test_kv_cache.py:8`, `tests/test_quantizer.py:14` — unused `math` / `pytest` [Low]
- Project-wide: no ruff/black/ty gate; `# ty:ignore` markers without corresponding CI [Medium]

---

## Validation Command Output

### Smoke import (`uv run python -c "import turboquant_mlx; ..."`)

```
Using CPython 3.12.13
Creating virtual environment at: .venv
Installed 40 packages in 142ms
import OK
```

### `uv run pytest -q`

```
..........                                                               [100%]
10 passed in 1.08s
```

### `uvx ruff check src tests scripts proof.py`

```
F841 Local variable `rng` is assigned to but never used
  --> src/turboquant_mlx/codebook.py:49:5

F401 [*] `typing.Optional` imported but unused
  --> src/turboquant_mlx/kv_cache.py:15:20

F841 Local variable `K` is assigned to but never used
   --> src/turboquant_mlx/kv_cache.py:177:9

F401 [*] `dataclasses.field` imported but unused
  --> src/turboquant_mlx/quantizer.py:10:36

F841 Local variable `T` is assigned to but never used
   --> src/turboquant_mlx/quantizer.py:182:9

F401 [*] `math` imported but unused
  --> src/turboquant_mlx/rotation.py:9:8

F401 [*] `math` imported but unused
  --> tests/test_kv_cache.py:8:8

F401 [*] `pytest` imported but unused
  --> tests/test_kv_cache.py:12:8

F401 [*] `pytest` imported but unused
  --> tests/test_quantizer.py:14:8

F841 Local variable `ck` is assigned to but never used
  --> tests/test_quantizer.py:63:9

Found 10 errors.
[*] 6 fixable with the `--fix` option
```

### Docstring coverage (custom inspect script)

```
undocumented: turboquant_mlx.codebook.cli_main
total public: 23, undocumented: 1
```

### `python -m py_compile` on all modules

```
compile OK
```
