# Improvements Checklist

**Generated from review:** _dev/reviews/001/critical_analysis.md
**Date:** 2026-05-04

---

## P0 — Blockers

- [ ] **[CI/CD]** Add a `.github/workflows/ci.yml` that runs `pytest`, `ruff check`, and a smoke `tq-generate --help` on every push (macOS-14 runner has Apple Silicon for the real path) — `<repo>/.github/workflows/ci.yml` — Effort: M
- [ ] **[Testing]** Add an end-to-end integration test that patches a tiny dummy mlx-lm-shaped model and asserts decode logits match baseline within tolerance — would catch the recent class of bugs (RoPE, scale, decode-flip, ingestion) — `tests/test_e2e_decode.py` — Effort: L
- [ ] **[Safety]** Add a `pack_bits`/`unpack_bits` round-trip test parameterised over `bits ∈ {2,3,4}` and `D ∈ {16, 17, 64, 128, 160, 256}` — `tests/test_codebook.py` — Effort: S
- [ ] **[Safety]** Replace bare `assert k_buf is not None and v_buf is not None` (`kv_cache.py:132`) with a real check or eliminate via type narrowing; assertion vanishes under `-O` — `src/turboquant_mlx/kv_cache.py:132` — Effort: S

## P1 — Pre-release

- [ ] **[API]** Make `TurboQuantLayerCache.scale` a required parameter — silent default of `1/sqrt(D)` produced gibberish on gemma4 last round — `src/turboquant_mlx/kv_cache.py:59` — Effort: S
- [ ] **[Safety]** Raise (don't silently no-op) when prefill cannot extract `kv_pair` — `src/turboquant_mlx/patch.py:91-99` — Effort: S
- [ ] **[Error Handling]** Validate JSON codebook payloads on load (length, dtype, monotonicity) — `src/turboquant_mlx/codebook.py:114-126` — Effort: S
- [ ] **[Error Handling]** Replace bare `except (AttributeError, TypeError): pass` cache-offset write with explicit capability probe — `src/turboquant_mlx/patch.py:174-178` — Effort: S
- [ ] **[Testing]** Add GQA tests with `n_kv_heads ∈ {2, 4}` to `test_kv_cache.py` — `tests/test_kv_cache.py` — Effort: M
- [ ] **[Testing]** Pin `quantize_with_codebook` boundary behaviour with a regression test (the off-by-one fix at `codebook.py:155` had no test) — `tests/test_codebook.py` — Effort: S
- [ ] **[Conventions]** Add `[tool.ruff]` config to `pyproject.toml` and gate CI on `ruff check` — Effort: S
- [ ] **[Docs]** Update `README.md:138-141` "Limitations" — numpy CPU path is gone, prefill behaviour is correct — `README.md` — Effort: S
- [ ] **[Docs]** Reconcile `head_dim` advice (160 in README vs. 256/512 in `kv_cache.py:333`) — `README.md`, `structure.md` — Effort: S
- [ ] **[Docs]** Drop or replace the placeholder `gemma-4-26B-A4B Q8_0` reference in `proof.py:18, 95` — Effort: S

## P2 — Should-fix

- [ ] **[Performance]** Add a `pytest-benchmark` suite covering compress, decompress, attention(buffer-only), attention(mixed) — without this every perf change is a guess — `tests/bench/` — Effort: M
- [ ] **[Performance]** Investigate caching the float32 one-hot bridge in `_scatter_or_row` per `(bits, D)` — currently rebuilt every `pack_bits` call — `src/turboquant_mlx/codebook.py:316-319` — Effort: S
- [ ] **[API]** Either remove the unused QJL path from the decode hot path or expose a `use_qjl=False` knob to skip its computation at compress time — `src/turboquant_mlx/quantizer.py` — Effort: M
- [ ] **[Conventions]** Fix all 10 ruff F401/F841 findings — Effort: S
- [ ] **[Conventions]** Either wire up the `ty` type checker (referenced by `# ty:ignore` markers) in CI, or remove the markers — `proof.py`, `scripts/`, `cli/` — Effort: M
- [ ] **[Deps]** Remove the `[server]` extra from `pyproject.toml` until a server module actually exists — `pyproject.toml:20-23` — Effort: S
- [ ] **[Deps]** Reconcile dependency version claims between `pyproject.toml` and `structure.md:213-218` — Effort: S
- [ ] **[Docs]** Add a CHANGELOG.md and start tagging `v0.2.0` — Effort: S
- [ ] **[Safety]** Add NaN/Inf guard at `kv_cache.attention()` exit (one debug-only check) — Effort: S
- [ ] **[CI/CD]** Add `pip-audit` / dependency vulnerability scan to CI — Effort: S

## P3 — Nice-to-have

- [ ] **[Performance]** Scaffold a Metal kernel for `score(query, packed_keys) → scores` to remove the per-step decompress on the value path — only this can make TQ outperform baseline tok/sec on Apple Silicon — Effort: L
- [ ] **[API]** Add a `get_wrappers(model)` accessor so callers don't have to capture `patch_model`'s return value — `src/turboquant_mlx/patch.py` — Effort: S
- [ ] **[Conventions]** Make module-level codebook caches thread-safe (or document that they're not) — `src/turboquant_mlx/codebook.py:90, 183-184` — Effort: S
- [ ] **[Performance]** Pre-allocate a fixed-size buffer of `(buffer_size + flush_batch, K*D)` and use index-based writes instead of growing chunk lists — `src/turboquant_mlx/kv_cache.py:107-123` — Effort: M
- [ ] **[Docs]** Add an `examples/` directory with a runnable end-to-end script that prints a baseline-vs-TQ comparison table — Effort: S
- [ ] **[Conventions]** Honour the `rng_seed` parameter in `lloyd_max` or remove it — `src/turboquant_mlx/codebook.py:49` — Effort: S

---

## Progress

**Total items:** 26
**P0:** 4 | **P1:** 11 | **P2:** 10 | **P3:** 6
