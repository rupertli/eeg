## Frequency-band interaction
It learns how combinations of EEG frequency bands jointly contribute to emotion recognition—for example, whether alpha and beta activity together provide more information than either band alone.

## Temporal augmentation
It creates varied training sequences by adding small noise, shifting features in time, or slightly scaling their amplitude. This improves robustness and reduces overfitting while leaving validation and test data unchanged.

## Squeeze-and-Excitation (SE)
Squeeze-and-Excitation (SE) is an attention mechanism that adaptively reweights feature channels:
- Squeeze: Aggregate each feature channel, commonly using global average pooling, to obtain a compact summary.
- Excitation: Pass the summary through a small neural network and a sigmoid function to learn an importance weight for each channel.
- Recalibration: Multiply the original features by these weights, emphasizing informative channels and reducing less useful ones.

  
In the EEG model, SE is applied to the five frequency bands, so it learns the relative importance of delta, theta, alpha, beta, and gamma features.
