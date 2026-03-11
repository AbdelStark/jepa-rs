# specs/gherkin/collapse_prevention.feature

Feature: Collapse Prevention
  JEPA training must prevent representational collapse where all inputs
  map to the same output vector. This is the primary failure mode.

  Background:
    Given a VICReg regularizer with default parameters
    And a batch size of 32
    And an embedding dimension of 256

  Scenario: Collapsed representations produce high variance loss
    Given two identical constant tensors of shape [32, 256]
    When I compute the VICReg loss
    Then the variance term should be greater than 10.0
    And the covariance term should be 0.0
    And the invariance term should be 0.0

  Scenario: Diverse representations produce low variance loss
    Given two tensors with per-dimension standard deviation above 1.0
    When I compute the VICReg loss
    Then the variance term should be approximately 0.0

  Scenario: Correlated dimensions produce high covariance loss
    Given a tensor where dimensions 0 and 1 are perfectly correlated
    When I compute the VICReg loss
    Then the covariance term should be greater than 0.0

  Scenario: Orthogonal dimensions produce low covariance loss
    Given a tensor where all dimensions are orthogonal
    When I compute the VICReg loss
    Then the covariance term should be approximately 0.0


# specs/gherkin/energy.feature

Feature: Energy Functions
  Energy functions measure compatibility between predicted and actual
  representations. Lower energy means better prediction.

  Scenario: L2 energy between identical representations is zero
    Given two identical representation tensors
    When I compute the L2 energy
    Then the result should be less than 1e-6

  Scenario: L2 energy is always non-negative
    Given two random representation tensors
    When I compute the L2 energy
    Then the result should be greater than or equal to 0.0

  Scenario: L2 energy is symmetric
    Given representation A and representation B
    When I compute energy(A, B) and energy(B, A)
    Then both values should be equal within tolerance 1e-6

  Scenario: Cosine energy between parallel vectors is zero
    Given two representation tensors pointing in the same direction
    When I compute the cosine energy
    Then the result should be less than 1e-6

  Scenario: Cosine energy between orthogonal vectors is maximal
    Given two orthogonal representation tensors
    When I compute the cosine energy
    Then the result should be approximately 1.0


# specs/gherkin/ema.feature

Feature: Exponential Moving Average
  The target encoder is updated via EMA of the context encoder.
  This asymmetry prevents collapse.

  Scenario: EMA with momentum 1.0 keeps target unchanged
    Given a target weight of 5.0 and an online weight of 10.0
    When I apply EMA with momentum 1.0
    Then the target weight should remain 5.0

  Scenario: EMA with momentum 0.0 copies online directly
    Given a target weight of 5.0 and an online weight of 10.0
    When I apply EMA with momentum 0.0
    Then the target weight should be 10.0

  Scenario: EMA with typical momentum moves target toward online
    Given a target weight of 0.0 and an online weight of 1.0
    When I apply EMA with momentum 0.996
    Then the target weight should be approximately 0.004

  Scenario: EMA converges to online weights after many steps
    Given a target weight of 0.0 and a constant online weight of 1.0
    When I apply EMA with momentum 0.99 for 1000 steps
    Then the target weight should be within 0.01 of 1.0

  Scenario: Cosine momentum schedule increases over training
    Given a cosine momentum schedule from 0.996 to 1.0 over 10000 steps
    When I query momentum at step 0
    Then it should be 0.996
    When I query momentum at step 5000
    Then it should be approximately 0.998
    When I query momentum at step 9999
    Then it should be approximately 1.0


# specs/gherkin/masking.feature

Feature: Masking Strategies
  Masking determines which parts of the input are context (visible)
  and which are targets (to predict). This is the most critical
  design decision in JEPA.

  Scenario: Block masking partitions all patches
    Given a 14x14 patch grid (196 patches total)
    And a block masking strategy with 4 target blocks
    When I generate a mask
    Then context_indices + target_indices should cover all 196 patches
    And there should be no overlap between context and target indices

  Scenario: Block masking respects scale constraints
    Given a 14x14 patch grid
    And target_scale range of (0.15, 0.2)
    When I generate 100 masks
    Then each mask should have between 29 and 39 target patches (15-20% of 196)

  Scenario: Spatiotemporal masking spans both space and time
    Given a video tokenized into 8 frames of 14x14 patches
    And a spatiotemporal masking strategy with temporal_extent (2, 4)
    When I generate a mask
    Then each target region should span at least 2 consecutive frames
    And each target region should be spatially contiguous

  Scenario: Masking is reproducible with same seed
    Given any masking strategy and a fixed random seed
    When I generate a mask twice with the same seed
    Then both masks should be identical

  Scenario: Masking is different with different seeds
    Given any masking strategy
    When I generate masks with seed 42 and seed 43
    Then the masks should differ


# specs/gherkin/world_model.feature

Feature: World Model Planning
  An action-conditioned world model predicts future states given actions,
  enabling an agent to plan before acting.

  Scenario: Single-step prediction produces valid representation
    Given a trained world model
    And a current state representation of shape [1, 16, 256]
    And an action of shape [1, 4]
    When I predict the next state
    Then the result should have shape [1, 16, 256]
    And the result should not be all zeros

  Scenario: Multi-step rollout produces correct trajectory length
    Given a trained world model
    And an initial state
    And a sequence of 10 actions
    When I perform a rollout
    Then I should get exactly 11 states (initial + 10 predicted)

  Scenario: Better plans have lower cost
    Given a world model and a cost function
    And a start state far from a goal state
    And plan A that moves toward the goal
    And plan B that moves away from the goal
    When I evaluate both plans
    Then plan A should have lower cost than plan B

  Scenario: Planning finds a path to goal
    Given a world model trained on a 2D navigation environment
    And a start position and a goal position
    When I run gradient-based planning for 100 iterations
    Then the final plan cost should be lower than the initial plan cost


# specs/gherkin/checkpoint.feature

Feature: PyTorch Checkpoint Loading
  jepa-rs must be able to load weights from the reference Python
  implementations to enable transfer learning and differential testing.

  Scenario: Load I-JEPA weights from safetensors
    Given an I-JEPA ViT-H/14 checkpoint in safetensors format
    When I load it into a jepa-rs VitEncoder
    Then all weight tensors should have matching shapes
    And a forward pass should produce non-zero output

  Scenario: Loaded model matches Python output
    Given a known input tensor and expected output from Python
    And an I-JEPA checkpoint loaded into jepa-rs
    When I run a forward pass on the known input
    Then the output should match the Python reference within tolerance 1e-4

  Scenario: Load V-JEPA 2 weights
    Given a V-JEPA 2 ViT-g checkpoint in safetensors format
    When I load it into a jepa-rs VitVideoEncoder
    Then all weight tensors should have matching shapes
    And the 3D-RoPE positional encoding should be correctly initialized
