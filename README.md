### Aim of this work: Establish an accurate multi-point force reconstruction, incl.: 
- Z-axis normal force 
- X, Y axis shear force 
- Z-axis moment

Objectives:
1) Using DBSCAN clustering algorithm to capture the marker points, measure the cluster size as intensity and obtain a z-axis deformation map from intensity interpolation -> finished
2) From the deformation map, evaluate the local maxima -> finished
3) The static stress can be obtained by correlating the marker displacement with the force applied on x, y plane. Thus, the objective is to evaluate the markers displacement with respect to their original relax state. It can be addressed as a non-rigid robust point set registration problem, which is researched by image processing community. But the computational complexity is very high solving such optimization problem. CPD algorithm was tried; it is running at ~30 FPS by a 12-core CPU running at 5.0GHz. -> In progress

<img width="2122" height="776" alt="image" src="https://github.com/user-attachments/assets/c6f043bf-d268-47a9-a33a-986e76a09808" />*Demonstration of this work*

The CPD library we use here is contributed by Gatti et al. in their paper `Gatti et al., (2022). PyCPD: Pure NumPy Implementation of the Coherent Point Drift Algorithm. Journal of Open Source Software, 7(80), 4681, https://doi.org/10.21105/joss.04681` and GitHub Repo: https://github.com/siavashk/pycpd

