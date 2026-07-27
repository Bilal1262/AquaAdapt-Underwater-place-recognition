# Architecture and objective

AquaAdapt maps a visible underwater image to a compact unit vector for exact cosine place
retrieval. The official DINOv2 ViT-S/14 produces a 384-dimensional CLS token. The learned
head is

```text
Linear(384, 512) → GELU → Dropout → Linear(512, 256) → L2 normalize.
```

The backbone stays in evaluation mode when entirely frozen. Projection-only training and
unfreezing the final one or two transformer blocks are supported. Head and backbone
parameters use separate AdamW learning rates.

## Multi-positive InfoNCE

Let normalized descriptors be \(z_i\), cosine similarity \(s_{ij}=z_i^\top z_j\),
temperature \(\tau\), and valid positive indices \(P(i)\). Self-similarity is removed:

\[
\mathcal{L}_i =
-\log \frac{\sum_{p \in P(i)} \exp(s_{ip}/\tau)}
{\sum_{a \ne i} \exp(s_{ia}/\tau)}.
\]

The implementation uses `logsumexp` for numerator and denominator. Anchors without a
positive are omitted; a fully empty positive mask returns a differentiable zero. The core
training batch always supplies the other augmented view of the same frame. Temporal or
geometric positives can only come from the training split and must satisfy configured
time and pose-distance gates.

