# Tactle-model-BioTacTip-multipoint
The code repository for paper:An Interpretable Tactile Model for Multi-Point 3D Contact Force Reconstruction Using BioTacTip

main.py----including image processing, DBSCAN, visualization(try with camera or video “multi-point_contact.avi”)
heat_map.py----show the force depth(try with camera or video “multi-point_contact.avi”)
math_relation.py----for single and two point contact, derive the reconstruction equation(dataset: intensity_force_single.csv, intensity_force_two.csv)
force_prediction1.py----predict single-point force from real-time intensity
force_prediction2.py----predict multi-point force from real-time intensity
realtime_line.py----prediction results(try with"multi-point_prediction.csv" and "normal_shear_prediction.csv")

