Phase 5: Hard Negative Sampling Introduction
Motivation

Baseline performance using random negative sampling:

Validation AUC ≈ 0.95

Validation AP ≈ 0.94

Loss ≈ 0.26

Observation:
Random negatives are structurally trivial. The model learns to separate obvious non-edges, inflating performance.

Objective:
Introduce structurally challenging negatives to improve discriminative capacity and obtain realistic evaluation metrics.

5.1 Replacement of Random Negative Sampling
Previous Strategy

Used PyTorch Geometric random negative sampling:

Uniform random node pairs

No structural constraints

Easy classification boundary

New Strategy

Implemented 2-hop hard negative sampling using NetworkX.

Definition:
A hard negative (u, v) satisfies:

(u, v) is not an edge

u and v share at least one common neighbor

Rationale:
These node pairs are structurally plausible future links and significantly harder to classify.

Implementation:

Created utils/negative_sampling.py

Implemented sample_hard_negatives(graph, num_samples)

Removed torch_geometric.utils.negative_sampling

Integrated sampler into training and validation loops

5.2 Decoder Upgrade
Previous Decoder

Dot product decoder.

Limitation:
Linear similarity measure.
Limited expressiveness.

New Decoder

MLP-based decoder.

Architecture:

Input: concatenated node embeddings

Hidden layer: 256 units

Dropout: 0.3

Output: single logit

Rationale:
Improves modeling of non-linear interactions between node embeddings.

5.3 Observed Training Behavior

After introducing hard negatives:

Early Epochs:

Loss ≈ 0.70

Validation AUC ≈ 0.60–0.65

Later Epochs (Epoch 50):

Loss ≈ 0.50–0.53

Validation AUC ≈ 0.82–0.83

Validation AP ≈ 0.81–0.83

Interpretation:
Performance decreased relative to random negative baseline because the task became significantly harder.

Important:
Lower AUC does not indicate worse model quality.
It reflects stronger evaluation protocol.

5.4 Identified Structural Issue

Current limitation:
Hard negatives are sampled from the full target snapshot, which still contains validation edges.

Consequence:
Potential structural leakage.
Model trains on a graph that includes edges later used for validation.

Planned correction:

Construct training graph by removing validation edges.

Sample hard negatives from training graph only.

Ensure strict separation between train and validation structure.

Current Architecture (Updated)

Community Features
↓
GraphSAGE Structural Encoder
↓
Temporal Transformer
↓
Temporal Node Embeddings
↓
MLP Decoder
↓
Binary Cross Entropy Loss
↓
Hard 2-Hop Negative Sampling

Research Insight

Key observation:
Performance with random negatives was artificially inflated.
Hard negative sampling provides more realistic and structurally meaningful evaluation.

Current AUC ≈ 0.83 with hard negatives is stronger than 0.95 with random negatives.