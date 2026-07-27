# Object-role contamination ledger — 2026-07-27

Status: frozen before new representation outcomes are opened.

## Purpose

This ledger records whether an object's learned representation outcome has
already influenced a recipe, threshold, training duration, checkpoint rule, or
diagnostic decision. Dataset rendering and geometry inspection are recorded
separately: they can influence corpus design, but they do not reveal whether a
learned keypoint recipe succeeds on that object.

## Frozen roles

| Object | Representation-outcome exposure | Dataset/geometry exposure | Frozen role |
|---|---|---|---|
| `engineers_hammer_vray` | Extensive: 324-run sweep, Tasks 20/55/80, Gate 0, Decision 2.3 | Yes | development |
| `b03_banana_01_high` | No versioned trained-representation result found or used for recipe choice | Yes: rendering/visibility/generation checks | confirmation |
| `kettle` | No versioned trained-representation result found or used for recipe choice | Yes: rendering/visibility/generation checks | confirmation |
| `dewalt_compact_drill_vray` | No versioned trained-representation result found or used for recipe choice | Yes: rendering/visibility/generation checks | final test |
| `b03_trumpet_vray` | No versioned trained-representation result found or used for recipe choice | Yes: rendering/visibility/generation checks | final test |
| `toy_monkey_medium` | No versioned trained-representation result found or used for recipe choice | Yes: rendering/visibility/generation checks | final test |

## Evidence basis and limitation

- Versioned training, sweep, diagnostic, Gate 0, Decision 2.3, and checkpoint
  paths in the isolated branch name only `engineers_hammer_vray`.
- The other five identities appear in dataset manifests, generators, candidate
  audits, and rendering/visibility design documents.
- This is a repository-evidence audit, not proof that no unrecorded human
  observation ever occurred.

If any learned non-hammer result is later found to have influenced the recipe,
thresholds, epoch rule, or checkpoint policy, stop before opening new outcomes
and amend the role assignment.

## Confirmation burn rule

Confirmation outcomes may reject the frozen recipe. They may not tune it while
remaining confirmation evidence. If banana or kettle causes a recipe change,
that object is thereafter development evidence. The three final-test objects
remain sealed until the resulting confirmation decision is frozen.
