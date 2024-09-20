#draw the real-time force depth map by processing the raw video


import cv2 as cv
import numpy as np
import time
from scipy.spatial import ConvexHull, QhullError
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

# Mask and area limitation
radius = 240
center = (180, 240)
video_file_path = 'multi-original_processed.avi'  # Replace with your video file path
cap = cv.VideoCapture(video_file_path)

fps = cap.get(cv.CAP_PROP_FPS)
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for MP4
out_heatmap = cv.VideoWriter('heatmap_multi.avi', fourcc, fps, (350, 350), True)  # False for grayscale


def area_to_depth(area):
    return 0.6736 * np.power(area, 0.3506)

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    mask = np.zeros((350, 350), dtype=np.uint8)
    cv.circle(mask, center, radius, (255), -1)
    circular_cropped = cv.bitwise_and(gray_frame, gray_frame, mask=mask)
    dst2 = cv.medianBlur(circular_cropped, 13)
    cv.imshow('',dst2)

    coordinates = np.column_stack(np.where(dst2 > 50))
    intensities = dst2[dst2 > 50]

    if len(coordinates) > 0:
        db = DBSCAN(eps=70, min_samples=3).fit(coordinates)
        labels = db.labels_
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f'Number of clusters: {num_clusters}')
        heatmap_bg = np.zeros_like(dst2, dtype=np.float32)

        for k in set(labels):
            if k == -1:
                continue

            class_member_mask = (labels == k)
            cluster_points = coordinates[class_member_mask]

            cluster_img = np.zeros_like(dst2)
            for x, y in cluster_points:
                cluster_img[x, y] = dst2[x, y]

            _, binary_image = cv.threshold(cluster_img, 50, 255, cv.THRESH_BINARY)
            contours, _ = cv.findContours(binary_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if not contours:
                print("No contours found")
                continue

            depths = []
            centroids = []
            for contour in contours:
                area = cv.contourArea(contour)
                depth = area_to_depth(area)  # Convert area to depth
                if depth > 0:
                    M = cv.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        centroids.append((cx, cy))
                        depths.append(depth)  # Store depth

            sorted_indices = np.argsort(depths)[::-1]
            sorted_depths = np.array(depths)[sorted_indices]
            sorted_centroids = np.array(centroids)[sorted_indices]

            all_points = np.vstack([contour.reshape(-1, 2) for contour in contours])

            overlay = np.zeros_like(cluster_img, dtype=np.uint8)

            try:
                hull = ConvexHull(all_points)
                hull_points = all_points[hull.vertices]
                hull_points = np.array([[int(x), int(y)] for x, y in hull_points], np.int32)
                hull_points = hull_points.reshape((-1, 1, 2))

                cv.fillPoly(overlay, [hull_points], (255, 255, 255))

                overlay_mask = overlay > 0

                x = np.linspace(0, cluster_img.shape[1] - 1, cluster_img.shape[1])
                y = np.linspace(0, cluster_img.shape[0] - 1, cluster_img.shape[0])
                grid_x, grid_y = np.meshgrid(x, y)

                points = np.array(sorted_centroids)
                values = np.array(sorted_depths)

                grid_z = griddata(points, values, (grid_x, grid_y), method='cubic', fill_value=values.min())
                grid_z = np.nan_to_num(grid_z)

                heatmap_data_masked = grid_z.copy()
                heatmap_data_masked[~overlay_mask] = 0

                smooth_heatmap = gaussian_filter(heatmap_data_masked, sigma=20)
                smooth_heatmap = smooth_heatmap / 7
                smooth_heatmap[smooth_heatmap > 1] = 1

                heatmap_bg += smooth_heatmap

            except QhullError as e:
                print(f"Error in ConvexHull computation: {e}")

        heatmap_bg = heatmap_bg if num_clusters > 0 else np.zeros_like(dst2, dtype=np.float32)

        heatmap_final = cv.applyColorMap((heatmap_bg * 255).astype(np.uint8), cv.COLORMAP_JET)

        cv.imshow('Heatmap of Deformation', heatmap_final)

        # Write the frame to the heatmap video
        out_heatmap.write(heatmap_final)

    if cv.waitKey(1) & 0xFF == ord('s'):
        break

# Release resources
cap.release()
out_heatmap.release()
cv.destroyAllWindows()
