# Implementation Plan: Add ML Learning Package

## Steps

### Step 1: Temperature fix in MCTSEngine

Isolated change to existing code. Testable immediately by running the existing TicTacToe example — nothing should break, and you can verify temperature > 0 produces varied move selection. Good low-risk warm-up.

**Manual test:** Run the existing TicTacToe example and confirm nothing regresses. Verify that temperature > 0 produces varied move selection.

### Step 2: NeuralNetworkEvaluator base class + TicTacToeMLP + TicTacToeNNEvaluator

The entire forward-pass pipeline, but with random (untrained) weights. Wire it up in tictactoe_learning/main.py as a playable agent. Random weights means terrible play, but the plumbing runs — you can verify the position encoding, forward pass, and policy decoding all work before touching any training code. This is the most important seam to validate early.

**Manual test:** Run `main.py` with the neural agent. Play will be poor (random weights) — the goal is confirming the pipeline runs end-to-end without error.

### Step 3: TrainingSample + SelfPlayCollector

Generate training data without training anything yet. Testable by running self-play and printing sample statistics — value distribution, number of samples, policy shapes. Confirms data collection is correct before gradient descent enters the picture.

**Manual test:** Run self-play and print summary statistics — total samples collected, value distribution, mean policy entropy. Confirms data collection is correct before gradient descent enters the picture.

### Step 4: TrainingLoop + train.py

The gradient descent loop. Run train.py, watch loss decrease, save weights, then play against the trained agent in main.py and observe improved play.

**Manual test:** Run `train.py` and observe loss decreasing. Then run `main.py` with `neural` vs `random` and observe the trained agent playing more purposefully than the random-weights version from step 2.
