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

## Attempt 2 — Slurm job 53631881 — failed during restore verification

- Commit: `5262185801e691069034d92c7316707626f36f8d`
- Cluster: Lichtenberg, node `ghqd0001`
- State: `FAILED`, exit `1:0`, elapsed `00:00:17`
- Progress: the prelaunch lock and Arm A smoke epochs 1 and 2 completed.
  The safe checkpoint load succeeded. Arms B/C and the D1 summarizer did not
  run; D2 was not submitted.
- Test policy: no test finalization command ran.

Failure:

```text
TypeError: RNG state must be a torch.ByteTensor
```

Read-only inspection of the saved checkpoint showed:

```text
loader_generator_state: torch.Tensor, dtype=torch.uint8
torch_rng_state: torch.Tensor
runtime.torch: str
```

Cause: `restore_checkpoint()` loaded the entire checkpoint with
`map_location=device`. On a CUDA smoke job this moved the CPU loader-generator
state to a CUDA ByteTensor; `torch.Generator()` is a CPU generator and rejects
that state.

Bounded correction: load the checkpoint on CPU, let model/optimizer restoration
move their own state to parameter devices, and explicitly keep loader, Torch,
and CUDA RNG-state tensors on CPU for the RNG restoration APIs. A regression
test passes a CUDA target device without requiring CUDA execution and asserts
that checkpoint loading remains CPU-mapped and the loader-generator state is
restored exactly.

Read-only Fable 5 High verdict:

> PASS FIX. Loading on CPU is safe for model and optimizer because
> `nn.Module.load_state_dict` copies into existing device-resident parameters
> and `Optimizer.load_state_dict` casts state tensors to each parameter's
> device. CUDA RNG handling is semantically correct because both
> `get_rng_state_all` and `set_rng_state_all` use CPU ByteTensors. The
> regression test meaningfully pins `map_location="cpu"` and exact generator
> restoration. A different visible-GPU count at resume would remain a risk;
> Decision 2.3 requests exactly one visible GPU for every training task.

## Attempt 3 — Slurm job 53632077 — passed

- Commit: `490bffac39f67cca2a8f2d0f363ea413c0ec6fb0`
- Cluster: Lichtenberg, node `ghqd0001`
- State: `COMPLETED`, exit `0:0`, elapsed `00:00:37`
- Hardware: one NVIDIA H100 80GB HBM3
- Scope: Arm A, B, and C sequentially for seed 42, exactly two epochs each
- Result: forward, backward, optimizer, epoch-0/1/2 validation, gradient audit,
  checkpoint save/restore, source/provenance binding, and no-test-access
  assertions all passed
- D1 report:
  `decision23_results/DECISION23_D1_SMOKE_REPORT.json`
- D1 report SHA-256:
  `dd018c1609017cfbe94d08a96297a7673fb818464391eb4c919fd7faefc3f671`
- Test policy: the report records `scientific_result=false`, and every arm
  records `test_evaluated=false`

The exact D1 report then received a read-only Fable 5 High `PASS D1` review.
Only after this successful attempt were the frozen D2 arm-by-seed runs
submitted.

The two failed attempts carry no scientific result and never authorized D2.
They are retained here because they exposed and closed two checkpoint-resume
failure modes before the decision-bearing matrix.
