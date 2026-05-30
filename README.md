# TactMorph: Towards General Solution for the Marker Displacement Problem

## **Introduction** 
We implemented the learning-based image registration algorithm, [VoxelMorph](https://www.mit.edu/~adalca/files/papers/tmi2019_voxelmorph.pdf) (referred to as TactMorph in this repo), as a [Marker Displacement Method](https://doi.org/10.1109/JSEN.2023.3255861) in tactile perception. MDM aims to identify marker displacement between a pair of tactile images, where markers are at rest (or at their initial positions) and are moved under external force. Current MDM approaches are neither robust to marker aliasing nor able to handle occlusion, or tracking loss during marker matching (e.g. trivial matching). Instead of directly adopting the VoxelMorph algorithm, we downsample the input image pair to a very low resolution before feeding it into the U-Net model to prevent marker aliasing during registration. Experiment results showed that this small variation can effectively alleviate marker aliasing while preserving the reconstruction quality in the context of MDM.

![alt text](figures/aliasing.png)
