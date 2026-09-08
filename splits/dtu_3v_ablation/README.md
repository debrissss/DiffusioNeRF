# DTU scan41 3-view training-ID ablation

The two `common41` manifests are a controlled comparison. They use different
three-view training sets but the same validation IDs (`3, 4`) and the same 41
test IDs. The common test set is the reconstructed historical 43-view test set
with IDs `27` and `48` removed because they are training inputs in the
`5, 27, 48` arm.

`scan41_train_1_23_47_reconstructed_old43.json` preserves the reconstructed
historical split for protocol-internal comparison with the surviving old
scan41 renders. Its validation IDs are inferred because the original split
manifest is no longer present.
