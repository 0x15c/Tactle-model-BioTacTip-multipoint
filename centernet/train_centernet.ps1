 # Windows shell script
 conda activate GelSight-homemade
#  python train_centernet.py --images centernet_training_dataset\yolo_dataset\images\train --annotations centernet_training_dataset\yolo_dataset\labels\train --format yolo --val_images centernet_training_dataset\yolo_dataset\images\val --val_annotations centernet_training_dataset\yolo_dataset\labels\val --epochs 100 --batch_size 6 --project centernet --lr 1e-4 
python train_centernet.py --images .\centernet_training_dataset\yolo_new_sensor\images --annotations .\centernet_training_dataset\yolo_new_sensor\labels --format yolo --val_images .\centernet_training_dataset\yolo_new_sensor\images --val_annotations .\centernet_training_dataset\yolo_new_sensor\labels --epochs 50 --batch_size 4 --project centernet_new_sensor --lr 1e-4 

#   ..\dataset_new_sensor\marker_dataset