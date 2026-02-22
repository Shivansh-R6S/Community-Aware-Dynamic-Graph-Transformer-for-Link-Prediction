# Actions & Design Decisions Log
Community-Aware Dynamic Graph Transformer for Link Prediction

---

## 1. Dataset Selection
- Dataset: FB-Forum temporal interaction network
- Format: (source, target, timestamp)
- Nodes mapped to contiguous IDs starting from 0

Decision:
✔ Use FB-Forum as dynamic social network benchmark

---

## 2. Snapshot Construction
- Strategy: Rolling window + cumulative memory
- Window size: 30 days
- Step size: 15 days
- Memory: 60 days
- Total snapshots created: 11

Representation:
✔ Snapshots stored as NetworkX graphs (not PyG)

Reason:
- Cleaner integration with community detection
- Better research clarity
- Tensor conversion deferred to training stage

---

## 3. Community Modeling
- Algorithm: Louvain community detection
- Applied independently per snapshot

Extracted node-level features:
1. Community size
2. Intra-community degree
3. Inter-community degree

Feature shape per snapshot:
✔ (num_nodes, 3)

Decision:
✔ Community features used as explicit structural priors

---

## 4. Structural Encoder
- Model: GraphSAGE (custom implementation)
- Aggregator: Mean
- Input features:
  - Community features (3-dim)
  - Learnable node embeddings (32-dim)
- Architecture:
  - GraphSAGE Layer 1: 35 → 64
  - GraphSAGE Layer 2: 64 → 128

Output:
✔ Structural embeddings of shape (num_nodes, 128)

Reason:
- Stable on CPU
- Inductive capability
- Suitable for dynamic link prediction

---

## 5. Integration Validation
- End-to-end test:
  - Data loader → snapshots → community → GraphSAGE
- Test file:
  - training/test_structural_encoder.py

Result:
✔ Structural embedding generation successful
✔ Output shape verified: (899, 128)

---

## 6. Hardware Constraints
- Local system:
  - RAM: 8 GB
  - GPU: Not required
- Decision:
Structural modeling runs locally
No Colab needed for current pipeline

---

## 7. Next Planned Step
- Temporal modeling across snapshots
- Use Transformer encoder over snapshot embeddings
- Objective:
  - Dynamic link prediction
  - Capture temporal evolution of node relationships

Status:
Not started

## Phase 1: Temporal Modeling (Baseline)

✔ Built GraphSAGE structural encoder (128-dim output)
✔ Extracted community features (size, intra-degree, inter-degree)
✔ Integrated structural encoder + community features
✔ Designed and implemented Temporal Transformer
✔ Used learnable positional encoding
✔ Used 2-layer transformer encoder (4 heads, d_model=128)
✔ Verified shape flow:
    - Structural embeddings: (N, 128)
    - Stacked temporal input: (T-1, N, 128)
    - Final temporal embeddings: (N, 128)

Decision:
Use one-step ahead prediction protocol:
    Snapshots 1–10 → predict links in snapshot 11
