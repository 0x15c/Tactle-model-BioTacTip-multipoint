**Poster:**

<img width="6622" height="9362" alt="Poster_BioTacTip_Interpretable_Model" src="https://github.com/user-attachments/assets/5813227f-f933-47f2-ab42-563df7f5158f" />

### Aim of this work: Establish an accurate multi-point force reconstruction, based on BioTacTip, incl.: 
- Z-axis normal force 
- X, Y axis shear force 
- Z-axis moment

Next steps:
1) Calibrate the sensor with ground truth
2) Design grasping experiments with advantages of our reconstruction result
3) ~*Optional?* Running by CPU is too slow. Maybe the CPD part algorithm could be rewritten by CuPy.~ The GPU version is even slower.
4) Find if we can address the difficulty in terms of computing by implementing GNN.


The CPD library we use here is contributed by Gatti et al. in their paper `Gatti et al., (2022). PyCPD: Pure NumPy Implementation of the Coherent Point Drift Algorithm. Journal of Open Source Software, 7(80), 4681, https://doi.org/10.21105/joss.04681` and GitHub Repo: https://github.com/siavashk/pycpd

