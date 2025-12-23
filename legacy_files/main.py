"""process the initial video signal to get the results of multi-point clustering and force information"""

#realtime
import cv2 as cv
import numpy as np
import time
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull, QhullError
from scipy.interpolate import splprep, splev
import csv
from datetime import datetime
# Mask and area limitation
radius = 160 # Modification: 240 -> 160
center = (175, 175) # Modification: (180, 240) -> (175, 175)
# Above Modification is made due to the diff. preprocessing methods of video, compared to the author.


# Camera get
#cap = cv.VideoCapture(1)
#ret, frame = cap.read()
#time.sleep(1)
video_file_path = 'output.mp4'  # Replace with your video file path
cap = cv.VideoCapture(video_file_path)
if not cap.isOpened():
    print("Can't open the camera")
    exit()
fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for MP4
out_original = cv.VideoWriter('original_processed.avi', fourcc, 6.5, (350, 350), False)  # False for grayscale

# Define video writer for heatmap video
out_colormark = cv.VideoWriter('color_shear_marked.avi', fourcc, 6.5, (350, 350), True)

# Initialize window
cv.namedWindow('Result Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Result Image', 1250, 1550)

def initialize_result_image():
    result_image = np.zeros((1250, 1550, 3), dtype=np.uint8)

    # Title
    cv.putText(result_image, 'Filtered Image', (10, 480), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Clustered Image', (550, 480), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Heat Map', (1150, 480), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Cluster 1', (10, 800), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Cluster 2', (260, 800), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Cluster 3', (510, 800), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv.putText(result_image, 'Cluster 4', (760, 800), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result_image

result_image = initialize_result_image()
cv.imshow('Result Image', result_image)

# Create and open the CSV file
with open('intensity.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Time', 'Cluster1_Intensity', 'Cluster1_Centroid', 'Cluster1_Area',
                        'Cluster2_Intensity', 'Cluster2_Centroid', 'Cluster2_Area',
                        'Cluster3_Intensity', 'Cluster3_Centroid', 'Cluster3_Area'])  # CSV header

    # Read frame in circle and output the clustering results
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive the video frame")
            break
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #cut_frame = gray_frame[60:410, 145:495]  # Used for amcap video

        mask = np.zeros((350, 350), dtype=np.uint8)
        cv.circle(mask, center, radius, (255), -1)
        circular_cropped = cv.bitwise_and(gray_frame, gray_frame, mask=mask)
        dst2 = cv.medianBlur(circular_cropped, 13) # dst2 is median blurred original image
        out_original.write(dst2)
        coordinates = np.column_stack(np.where(dst2 > 50)) # Extracts destination pixel coordinate whose illuminance > 50
        intensities = dst2[dst2 > 50] # give the corresponding illum. val.

        # Density-based spatial clustering of application with noisess
        if len(coordinates) > 0:
            db = DBSCAN(eps=70, min_samples=3).fit(coordinates)
            labels = db.labels_
            # labels_ : ndarray of shape (n_samples)
            # Cluster labels for each point in the dataset given to fit().
            # Noisy samples are given the label -1.
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            # print(f'Number of clusters: {num_clusters}')

            if num_clusters < 1:
                print("No valid clusters found")
                cv.putText(result_image, 'No valid clusters found', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv.imshow('Result Image', result_image)
                if cv.waitKey(1) & 0xFF == ord('s'):
                    break
                continue

            # Use different colors to show results of clustering
            clustered_image = cv.cvtColor(dst2, cv.COLOR_GRAY2BGR) # Restore image from Y channel
            white_background = np.ones_like(clustered_image) * 255
            heatmap = np.ones_like(clustered_image) * 255
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]  # BGR format: blue, green, red, yellow
            cluster_info = []
            overlay = np.zeros_like(white_background, dtype=np.uint8)

            for k, col in zip(set(labels), colors): # Put labels on color
                if k == -1:
                    continue

                class_member_mask = (labels == k) # For each label k (i.e. the classification) pick up coordinates of those pts.
                cluster_points = coordinates[class_member_mask] # n by 2 arrray, n is # of clustered pts.
                centroids = np.mean(cluster_points, axis=0)
                x_c, y_c = int(centroids[0]), int(centroids[1])

                # DEBUG
                # cv.imshow("DEBUG", dst2)

                # Determine the region and set color
                if x_c < dst2.shape[0] // 2 and y_c < dst2.shape[1] // 2:
                    color = colors[2]  # Blue for left upper
                elif x_c < dst2.shape[0] // 2 and y_c >= dst2.shape[1] // 2:
                    color = colors[0]  # Red for right upper
                elif x_c >= dst2.shape[0] // 2 and y_c < dst2.shape[1] // 2:
                    color = colors[1]  # Green for left lower
                else:
                    color = colors[3]  # Yellow for right lower

                # Draw the centroid
                cv.circle(white_background, (y_c, x_c), 10, color, -1)
                if len(cluster_points) >= 3: # Draw the convex polygon
                    try:
                        hull = ConvexHull(cluster_points, qhull_options='QJ')
                        hull_points = np.array([[int(y), int(x)] for x, y in cluster_points[hull.vertices]], dtype=np.int32)
                        # hull_points1 = hull_points.reshape((-1, 1, 2)) ??????
                        # hull_points1 = hull_points.reshape(-1, 2)

                        # # Spline
                        tck, u = splprep([hull_points[:, 0], hull_points[:, 1]], k=2, s=4, per=True)
                        u_new = np.linspace(u.min(), u.max(), 50)
                        x_new, y_new = splev(u_new, tck, der=0)
                        # smooth_hull_points = np.vstack((x_new, y_new)).T
                        # smooth_hull_points1 = np.array([[(y), (x)] for y, x in smooth_hull_points])
                        # smooth_hull_points2 = smooth_hull_points1.reshape((-1, 1, 2))
                        # smooth_hull_points3 = smooth_hull_points2.astype(int)
                        smooth_hull_points = (
                                                np.vstack((x_new, y_new)).T
                                                .astype(float)
                                                .reshape(-1, 2)
                                                .astype(int))

                        cv.fillPoly(overlay, [smooth_hull_points], color)

                    except QhullError as e:
                        print(f"Error in ConvexHull computation: {e}")

                # Calculate intensity
                # @TODO Intensity can be normalized maybe
                cluster_intensity = np.sum(dst2[cluster_points[:, 0], cluster_points[:, 1]])
                cluster_info.append((color, cluster_intensity, (y_c, x_c)))

                # Calculate area
                cluster_mask = np.zeros_like(dst2, dtype=np.uint8)
                cluster_mask[cluster_points[:, 0], cluster_points[:, 1]] = 1 
                cluster_area = np.sum(cluster_mask)
                cluster_info[-1] += (cluster_area,)

            # Write data to CSV
            utc_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]  # Format UTC time as HH:MM:SS.sss
            row = [utc_time]
            for info in cluster_info:
                  row.extend([info[0], info[1], info[2]])
            csvwriter.writerow(row)

            # Code below will produce white_background, i.e. the second image in the 1st row
            cigema = 0.4
            cv.addWeighted(overlay, cigema, white_background, 1 - cigema, 0, white_background)
            out_colormark.write(white_background)

            result_image = initialize_result_image()
            filtered_resized = cv.resize(circular_cropped, (450, 450))
            white_background_resized = cv.resize(white_background, (450, 450))
            heatmap_resized = cv.resize(heatmap, (450, 450))
            if heatmap_resized.shape[2] == 4:
                heatmap_resized = cv.cvtColor(heatmap_resized, cv.COLOR_RGBA2BGR)

            result_image[:450, :450] = cv.cvtColor(filtered_resized, cv.COLOR_GRAY2BGR)
            result_image[:450, 550:1000] = white_background_resized
            result_image[:450, 1100:] = heatmap_resized

            text = f'Number of clusters: {num_clusters}'

            for i, (color, intensity, centroid, area) in enumerate(cluster_info):
                y_offset = 850 + i * 30
                x_offset = 10
                color_bgr = [int(c) for c in color]
                cv.circle(result_image, (x_offset, y_offset), 10, color_bgr, -1)
                info_text = f'Intensity: {intensity}, Centroid: {centroid}, Area: {area}'
                cv.putText(result_image, info_text, (x_offset + 20, y_offset + 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cluster_imgs = []
            cluster_intensities = []
            cluster_centroids = []
            cluster_area = []

            for k in set(labels): # labels: db_labels, i.e. DBSCAN labeling, e.g. pts labelled 0 are clustered in one set
                # use of set: count the # of classifications
                if k == -1:
                    continue

                cluster_mask = np.zeros_like(dst2, dtype=np.uint8)
                class_member_mask = (labels == k)
                cluster_points = coordinates[class_member_mask]

                # cluster_img = np.zeros_like(cv.cvtColor(dst2, cv.COLOR_GRAY2BGR))
                # cluster_img = np.zeros((350,350,3),np.uint8)
                # for x, y in cluster_points:
                cluster_img = cv.cvtColor(dst2, cv.COLOR_GRAY2BGR)
                
                cluster_img_resized = cv.resize(cluster_img, (220, 180))
                cluster_imgs.append(cluster_img_resized)

            empty_img = np.zeros((180, 220, 3), dtype=np.uint8)
            while len(cluster_imgs) < 4:
                cluster_imgs.append(empty_img)
            for i in range(4):
                x_offset = i * 250
                y_offset = 450
                result_image[y_offset:y_offset + 180, x_offset:x_offset + 220] = cluster_imgs[i]

            cv.imshow('Result Image', result_image)

        if cv.waitKey(1) & 0xFF == ord('s'):
            break

cap.release()
out_original.release()
out_colormark.release()
cv.destroyAllWindows()

