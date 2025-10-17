import numpy as np
import open3d as o3d
import matplotlib.cm as cm

# Example setup
n = 100
origins = np.random.rand(n, 3) * 2 - 1  # fixed origins in [-1,1]^3

def generate_vectors(n):
    # dummy vector generator (replace with your data stream)
    return np.random.randn(n, 3) * 0.3

# Function to create colored line set
def create_lines(origins, vectors, cmap=cm.viridis):
    points = np.vstack([origins, origins + vectors])
    
    # Lines connect i -> i+n
    lines = [[i, i + len(origins)] for i in range(len(origins))]
    
    # Color by vector length
    lengths = np.linalg.norm(vectors, axis=1)
    norm = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-9)
    colors = cmap(norm)[:, :3]  # take RGB from colormap
    
    # One color per line
    line_colors = [colors[i] for i in range(len(origins))]
    
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)
    return line_set

# --- Visualization loop ---
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initialize first frame
vectors = generate_vectors(n)
line_set = create_lines(origins, vectors)
vis.add_geometry(line_set)

while True:
    # Update with new vectors
    vectors = generate_vectors(n)
    new_line_set = create_lines(origins, vectors)

    line_set.points = new_line_set.points
    line_set.lines = new_line_set.lines
    line_set.colors = new_line_set.colors

    vis.update_geometry(line_set)
    vis.poll_events()
    vis.update_renderer()
