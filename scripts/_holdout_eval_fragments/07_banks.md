## 4. Train CE prototypes vs holdout eval sentence bank

**Train CE:** fixed templates → `PROTO_TRAIN[f]` (3 vectors).  
**Eval:** 5 held-out paraphrases per finding×class (strings never equal to train templates).  
At eval we sample one embedding per class (cached).
