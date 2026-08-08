# Architecture — Short Spoken Script (~2 min)

**Open:**
"The baseline showed that just subtracting two frozen embeddings can't read
change. So we keep MERLIN frozen and add one small trainable module that learns
the change between two scans and matches it to the comparison language in the
report."

**Frozen vs trainable (point at colors):**
"Everything blue is frozen — both MERLIN encoders. Only the two orange boxes
learn: a small cross-exam Transformer, and one projection layer. So we're
training a few million parameters, not the whole backbone."

**The key idea — antisymmetric difference:**
"Instead of naive subtraction, we compute the change as g(current, prior) minus
g(prior, current). Two wins from that: 'no change' becomes exactly the zero
vector — which fixes the baseline's biggest failure, that 'stable' had no
representation — and 'worse' and 'improved' become opposite signs of the same
axis, so if you swap the scans the answer just flips sign."

**How it learns:**
"We push this change embedding into MERLIN's shared space and use a contrastive
loss to pull it toward the report's change sentences. That's the only loss."

**The LLM splitter (bottom):**
