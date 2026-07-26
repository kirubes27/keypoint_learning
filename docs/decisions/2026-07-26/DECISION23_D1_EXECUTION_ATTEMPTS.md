# Decision 2.3 D1 execution attempts — 2026-07-26

## Attempt 1 — Slurm job 53631880 — failed before D1 completion

- Commit: `60d5c5d15760d578729093d52ef95e906acc23df`
- Cluster: Lichtenberg, node `ghqd0001`
- Accounting/partition assigned by submit plugin: `l0003029` / `acc_short`
- State: `FAILED`, exit `1:0`, elapsed `00:00:24`
- Progress: the prelaunch lock completed and Arm A completed smoke epochs 1
  and 2. Arms B/C and the D1 summarizer did not run.
- Test policy: no test finalization command was part of D1; D2 was not
  submitted.

Failure:

```text
_pickle.UnpicklingError: Weights only load failed.
Unsupported global: GLOBAL torch.torch_version.TorchVersion
```

Cause: on the cluster's PyTorch 2.5.1 environment, `torch.__version__` is a
`TorchVersion` string subclass. It was embedded in the checkpoint config as
that subclass, which the safe `weights_only=True` loader rejects.

Bounded correction: `runtime_identity()` converts all version values to
built-in strings before checkpoint serialization. A regression test performs
an in-memory `torch.save` followed by `torch.load(weights_only=True)` and
checks the restored runtime mapping.

Actual local evidence after correction: 50 tests passed, `py_compile` passed,
and `git diff --check` passed.

Read-only Fable 5 High verdict:

> PASS FIX
>
> The `str()` conversions target exactly the classes that break
> `weights_only=True`, while every remaining runtime field is already a
> built-in primitive or `None`. Frozen semantics are preserved because only
> value types change, not keys or values. The regression test exercises the
> actual failure path. Since D2 was never submitted and epochs 1-2 of the
> failed smoke carry no decision weight, rerunning D1 under the patched code
> introduces no frozen-programme violation.
