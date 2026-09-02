# AASIST implementation notes

This AASIST experiment follows the published AASIST design at a conceptual level: raw waveform input, a waveform front-end, spectro-temporal tokenization, graph attention over token nodes, and downstream classification using pooled node representations.

Important design choices:
- Input waveform length: 4.0 seconds at 16 kHz (64,000 samples), matching the standard fixed-length audio setup used in the project.
- Sampling rate: 16 kHz.
- Loss: weighted binary cross-entropy is used to account for the strong class imbalance in ASVspoof 2019 LA.
- Thresholding: the dev threshold is selected using the dev-only EER criterion and then frozen before official evaluation.
- This implementation is a faithful AASIST-style architecture, not a CNN masquerading as AASIST. It preserves the waveform-first, spectro-temporal attention structure rather than feeding mel features directly into a CNN.

The official evaluation set remains held out and is used only after the model and threshold are frozen.
