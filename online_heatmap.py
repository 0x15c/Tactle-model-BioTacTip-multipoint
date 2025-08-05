import numpy as np
import cv2 as cv
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import time
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
# Notice: image size is 350x350, which is the magic number in this script almost everywhere.

output_video_file_path = 'output_from_online_cap.mp4'
cap = cv.VideoCapture(0)
fps = cap.get(cv.CAP_PROP_FPS)
video_w, video_h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fourcc = cv.VideoWriter_fourcc(*'XVID')  # Codec for MP4

# crop and preprocessing
radius = 160
center = (175, 175)
cnt = [int(video_w/2),int(video_h/2)]
crop_px = 175
crop_py = 175
crop_offset_x = 0
crop_offset_y = -8
cropped_limits = [[cnt[0]-crop_px+crop_offset_x,cnt[1]-crop_py+crop_offset_y],[cnt[0]+crop_px+crop_offset_x,cnt[1]+crop_py+crop_offset_y]]
cropped_size = [2*crop_px, 2*crop_py]

# camera settings
cap.set(cv.CAP_PROP_FRAME_WIDTH, video_w)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, video_h)
cap.set(cv.CAP_PROP_EXPOSURE, -7.8)
cap.set(cv.CAP_PROP_BRIGHTNESS, 0)
cap.set(cv.CAP_PROP_CONTRAST, 64)
cap.set(cv.CAP_PROP_SATURATION, 60)
cap.set(cv.CAP_PROP_HUE, 0)
cap.set(cv.CAP_PROP_GAIN, 0)
# CAP_PROP_BRIGHTNESS: 0.0
# CAP_PROP_CONTRAST: 32.0
# CAP_PROP_SATURATION: 60.0
# CAP_PROP_HUE: 0.0
# CAP_PROP_GAIN: 0.0

# illuminance correct mask, makes darker at edge
def create_radial_mask(size, center=None, max_value=60, power=2):
    h, w = (size, size) if isinstance(size, int) else size
    cx, cy = (w//2, h//2) if center is None else center
    y, x = np.ogrid[:h, :w]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    normalized = distance / np.sqrt(cx**2 + cy**2)
    return np.clip((normalized ** power) * max_value, 0, max_value).astype(np.uint8)

o_ring_mask = create_radial_mask((350, 350), max_value=245)

# output video file settings
out_original = cv.VideoWriter('output_from_online_cap.mp4', fourcc, fps, cropped_size, False)
heatmap_output = cv.VideoWriter('output_from_online_cap.mp4', fourcc, fps, cropped_size, True)

frame_count = 0
frame_diff_gain = 1
frame_prev = np.zeros(cropped_size,np.uint8)
frame_blurred = np.zeros(cropped_size,np.uint8)
GaussianKrnlSize = (3,3)

# circular mask
c_mask = np.zeros((350, 350), dtype=np.uint8)
cv.circle(c_mask, center, radius, (255), -1)




'''
    a function takes DBSCAN results, returns a tuple (n, k, 2) where:
    n is the # of clusters
    k is the points count inside a cluster
    2 is the (x, y) coordinate
 '''
def dbscan_extractor(dbscan_result, points):
    labels = dbscan_result.labels_
    points = np.array(points)
    
    # Get unique cluster labels (excluding noise)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != -1]
    
    if len(unique_labels) == 0:
        return []
    
    cluster_info = []
    
    for cluster_id in unique_labels:
        cluster_mask = (labels == cluster_id)
        cluster_points = points[cluster_mask]
        cluster_info.append(cluster_points)  # Just append the actual points
    
    return cluster_info

# this function takes extracted cluster info as input, calculates centroids with intensity
def centroids_calc(cluster_array):
    result = np.zeros((0,2))
    intsty = np.zeros((0))
    for cluster in cluster_array:
        centroid = np.mean(cluster,axis=0)
        n_pts = cluster.shape[0]
        intensity = n_pts # there shall be some mapping...
        result = np.append(result,[centroid],axis=0)
        intsty = np.append(intsty,[intensity],axis=0)
    return result, intsty

def update_cluster_plot(cluster_array):
    """Update plot for real-time visualization"""

    plt.clf()  # Clear current figure
    plt.xlim(0,350)
    plt.ylim(0,350)
    plt.figure(figsize=(4, 4))
    if cluster_array.size == 0:
        plt.text(0.5, 0.5, 'No clusters found', 
                transform=plt.gca().transAxes, ha='center', va='center')
        plt.draw()
        plt.pause(0.001)
        return
    
    n_clusters = cluster_array.shape[0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    
    for i in range(n_clusters):
        cluster_points = cluster_array[i]
        valid_mask = ~np.isnan(cluster_points).any(axis=1)
        valid_points = cluster_points[valid_mask]
        
        if len(valid_points) > 0:
            plt.scatter(valid_points[:, 0], valid_points[:, 1], 
                       c=[colors[i]], s=30, alpha=0.7)
    
    plt.title(f'Real-time Clusters ({n_clusters} found)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.draw()
    plt.pause(0.001)

# mesh grid is used for interpolation
xspace = np.linspace(0, 350, 350)
yspace = np.linspace(0, 350, 350)
xgrid, ygrid = np.meshgrid(xspace, yspace)

# function for regulation of the interpolation result
def interpolate_reg(data, mode='zero-to-one',adaptive=False):
    if adaptive == True:
        min, max = np.min(data), np.max(data)
        data = data - min
        data = data*(1/(max-min))
        match mode:
            case 'cv-image':
                return (data*255).astype(np.uint8)
            case 'zero-to-one':
                return data
    else:
        match mode:
            case 'cv-image':
                data = data * 5.0 # scale factor
                data[data < 0] = 0
                data[data > 255] = 255
                return data.astype(np.uint8)
            case 'zero-to-one':
                return data # not finished yet
# fps init
time_start, time_end, fps = 0, 0, 0
# matplotlib settings
# plt.ion()
# fig, ax = plt.subplots(figsize=(4, 4))
time_0 = time.time() 
while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video stream or error reading frame.")
        break
    time_start = time.time()
    frame_cropped = frame[cropped_limits[0][1]:cropped_limits[1][1],cropped_limits[0][0]:cropped_limits[1][0]]
    cropped_frame = cv.bitwise_and(frame_cropped, frame_cropped, mask=c_mask)
    grey_frame = cv.cvtColor(cropped_frame,cv.COLOR_BGR2GRAY)
    grey_frame_corrected = cv.subtract(grey_frame,o_ring_mask)
    cv.imshow('grey_frame',grey_frame_corrected)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    # write capture
    # bgr_image = cv.cvtColor(grey_frame, cv.COLOR_GRAY2BGR)
    out_original.write(grey_frame_corrected)
    _,frame_binary = cv.threshold(grey_frame_corrected,100,255,cv.THRESH_BINARY)
    # cv.imshow('frame_binary',frame_binary)
    frame_blurred = cv.GaussianBlur(frame_binary,GaussianKrnlSize,0)

    # perform dbscan algorithm
    cluster_coordinates = cv.findNonZero(frame_binary).reshape(-1,2)
    cluster_coordinates[:,1]=350-cluster_coordinates[:,1] # mirroring
    cluster_data = DBSCAN(eps=2, min_samples=5).fit(cluster_coordinates)
    clusters = dbscan_extractor(cluster_data, cluster_coordinates)
    centroids, intensity = centroids_calc(clusters)
    # show cluster result
    # ax.clear()
    # plt.xlim(0,350)
    # plt.ylim(0,350)
    # ax.scatter(centroids[:,0],centroids[:,1],c=intensity, cmap='coolwarm', vmin=0, vmax=100)
    # plt.draw()
    # plt.pause(0.005)
    # sparsity extraction

    # intensity interpolation
    z_val = griddata(centroids, intensity, (xgrid, ygrid), method='linear',fill_value=0.0)
    reg_z = interpolate_reg(z_val,'cv-image')
    heatmap = cv.applyColorMap(gaussian_filter(reg_z,sigma=20), cv.COLORMAP_JET)
    heatmap = cv.flip(heatmap,0)
    timenow = time.time()-time_0
    cv.putText(heatmap, f"FPS={int(fps)}",(10,30),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(heatmap, f"T={float(timenow):.2f}",(10,60),cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)
    cv.imshow('heatmap',heatmap)
    heatmap_output.write(heatmap)
    # (try: frame diff)
    # if frame_count > 0:
    #     # Extended range because uint8 range is 0~255, by subtracting it will overflow
    #     frame_diff = np.abs(grey_frame_corrected.astype(np.int16) - frame_prev.astype(np.int16))*frame_diff_gain
    #     # Apply threshold to remove noise
    #     threshold_value = 30
    #     _, frame_diff = cv.threshold(frame_diff, threshold_value, 255, cv.THRESH_BINARY)
    #     frame_diff = frame_diff.astype(np.uint8)
    #     # cv.GaussianBlur(frame_diff,GaussianKrnlSize,0,frame_diff)
    #     cv.imshow('frame_diff',frame_diff)
    #     frame_prev = grey_frame_corrected
    # cv.imshow('frame_blurred',frame_blurred)
    time_end = time.time()
    frame_gen_time = time_end - time_start
    fps = 1/frame_gen_time
    frame_count+=1
cap.release()
out_original.release()
heatmap_output.release()
cv.destroyAllWindows()