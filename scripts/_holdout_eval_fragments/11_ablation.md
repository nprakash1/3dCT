## 6. Loss ablation under holdout-eval paradigm

Train CE always uses `PROTO_TRAIN`.  
**Tune early-stop and final test** always use **random holdout sentences** (1 of 5 per class).

Sweeps CE × Mag × SupCon (skip all-off). Results → Drive `ctclip_cache/ablations_holdout_eval/`.
