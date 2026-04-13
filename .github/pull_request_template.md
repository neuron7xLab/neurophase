<!-- PR template aligned with CLAUDE.md::response_contract.
     Fill in every section. Empty sections are a review blocker. -->

## RESULT

<!-- One or two sentences. What changed and why. No motivational
     filler. If a result is null, say "null" and commit it. -->

## FILES CHANGED

<!-- Short table or list. Name each file + one-line role. -->

## TESTS / GATES RUN

<!-- Paste the exact output of the four gates. Non-negotiable. -->

```
ruff check neurophase tests tools           : <result>
ruff format --check neurophase tests tools  : <result>
mypy --strict neurophase + tools            : <result>
pytest tests/ -q                            : <N passed, M skipped>
```

<!-- If you ran the acceptance suite or hardware suite, say so and
     paste the verdict. If not run, say "not run" and why. -->

## CONTRACTS ENFORCED

<!-- For every new invariant / guard / test added, name the contract
     it protects. If the PR only touches docs, say "docs only". -->

## LIMITATIONS

<!-- Honest list of what this PR does NOT do. What is still assumption
     rather than validation. What requires operator action downstream.
     No burying. -->

## EXACT REPRO COMMANDS

<!-- A reviewer must be able to reproduce your RESULT section by
     running these commands in order on a fresh clone. -->

```
<exact commands here>
```

---

## Classification (per CLAUDE.md::decision_logic)

Tick EXACTLY the classes this PR advances. Do not mix.

- [ ] REAL_BUG — fix in code + regression test
- [ ] MISSING_GUARD — invariant/guard + test
- [ ] DOC_DEBT — docs / claims / help text
- [ ] NON_ISSUE — analysis only, no code change

## Hard rules (per CLAUDE.md::hard_rules)

Confirm by ticking. A missing tick is a review blocker.

- [ ] No fake-live behavior. Replay is labelled replay.
- [ ] No claim inflation. Existence is not utility.
- [ ] No contract guessing — I read the code for every constant I touch.
- [ ] No decorative abstractions.
- [ ] No duplicate cores (replay and live still share one core).
- [ ] No silent drift — new external deps have a mechanical guard.
- [ ] No untested fixes — every `REAL_BUG` fix has a regression test.
- [ ] No "probably" — any un-run gate is said to be un-run.

## Stable-promotion gates touched (per docs/STABLE_PROMOTION.md)

- [ ] G1 Layer-D triplet  [ ] G2 calibration profile  [ ] G3 replay/live parity
- [ ] G4 tools in CI      [ ] G5 fresh-clone         [ ] G6 no open REAL_BUG
- [ ] G7 claims synced    [ ] G8 decision benchmark
- [ ] none of the above
