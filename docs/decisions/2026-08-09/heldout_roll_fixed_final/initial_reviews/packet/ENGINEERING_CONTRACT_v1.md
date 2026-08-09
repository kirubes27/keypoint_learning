# Held-out roll fixed-final engineering contract

Status: implementation contract only. This file does not approve a recipe,
epoch count, seed, object run, numerical threshold, or GPU launch.

## Plain-language purpose

For each confirmation or final-test object, train a fresh model using only the
frozen training frames. Save the final epoch model. Before any held-out image or
mask is decoded, write a receipt that fixes the model, configuration, history,
source commit, and approved run manifest. Then decode every unique held-out
image and mask once, run that saved model once over the in-memory frames, and
derive one test-stratum representation result from those outputs. Finally write
a second receipt binding the result.

“Provenance” here means the evidence needed to answer these concrete questions:

1. Which exact code, model file, configuration, seed, and epoch count produced
   the result?
2. Which exact training and test split files were used, and were their frame
   endpoints disjoint?
3. Which object-specific roll geometry and corpus inventory were used?
4. Were held-out contents decoded only after the model was final, and only in
   one evidence phase?
5. Did every reported representation metric come from the same saved model and
   the same in-memory inference?

## Must be true

- The run manifest is committed, hash-valid, and below the dedicated manifest
  directory.
- The decision specification, Pro review record, Fable review record, and user
  approval record referenced by that manifest are committed and hash-valid.
- The object role is `confirmation` or `final_test`.
- The transform is the frozen forward world-z roll stratum: +6 degrees, stride
  3, cyclic.
- The retained package is 512x512 input, 10 keypoints, 32 base channels,
  64-resolution heatmaps, shared affine operator, reflection padding, and no
  descriptor attachment loss.
- The recipe, seed, loss weights, and frozen epoch count equal the approved run
  manifest exactly. The code does not choose these values.
- Training and test indices are exact committed artifacts. Their selected frame
  endpoints are disjoint.
- The object has a separately registered and evidence-backed roll geometry
  binding. Hammer geometry cannot authorize banana, kettle, or another object.
- `final_model.pt`, `config.json`, and `history.json` exist before the pre-test
  receipt is written.
- The checkpoint is hashed and deserialized through the same file descriptor,
  and its embedded fixed-final contract and epoch are verified.
- Every unique test image and mask is hash-checked against the corpus inventory,
  decoded once, and retained in memory for the sole test inference.
- The evaluator receives test-only frame IDs and pair rows, logits, masks,
  physical states, the learned affine operator, and the object-specific
  projected centre.
- The evaluator reports descriptive representation evidence and explicitly does
  not make the scientific pass/fail decision.

## Must not happen

- No validation loader, best-checkpoint selection, per-object early stopping,
  second test opening, legacy whole-directory auto-evaluation, full-corpus
  diagnostic, automatic threshold decision, tuning, repair, or next-object
  launch is authorized by this path.
- No banana, kettle, final-test object, translation, scale, yaw, or pitch run is
  authorized merely because this implementation exists.

## Evidence files from one completed run

- `FIXED_FINAL_TRAINING_RECEIPT.json`: written after training and before held-out
  decoding.
- `fixed_final_test_bundle.json`: exact evaluator input.
- `fixed_final_test_representation_result.json`: test-stratum evidence.
- `FIXED_FINAL_EVIDENCE_RECEIPT.json`: binds the training receipt, checkpoint,
  bundle, result, and per-file test-opening ledger.

## Current stop condition

Engineering is ready only when the focused semantic tests and independent
implementation review pass. A real held-out run remains blocked until the user
approves the scientific decision specification and the selected object has a
registered roll geometry binding.
