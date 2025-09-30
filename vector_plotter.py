import numpy as np
import open3d as o3d
import matplotlib.cm as cm

class VectorFieldVisualizer:
    def __init__(self, origins, sphere_radius=3, cmap=cm.viridis):
        self.origins = origins
        self.sphere_radius = sphere_radius
        self.cmap = cmap
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        

        # Add fixed spheres at origins
        self.balls = self._create_origin_balls()
        for b in self.balls:
            self.vis.add_geometry(b)

        # Placeholder for arrows
        self.arrows = []

    def _create_origin_balls(self, color=[0.2, 0.2, 0.2]):
        balls = []
        for p in self.origins:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.sphere_radius)
            sphere.compute_vertex_normals()
            sphere.paint_uniform_color(color)
            sphere.translate(p)
            balls.append(sphere)
        return balls

    def _create_arrows(self, vectors, cyl_radius=1, cone_radius=3):
        arrows = []
        lengths = np.linalg.norm(vectors, axis=1)
        norm = (lengths - lengths.min()) / (np.ptp(lengths) + 1e-9)
        colors = self.cmap(norm)[:, :3]

        for i, vec in enumerate(vectors):
            p0 = self.origins[i]
            length = np.linalg.norm(vec)
            if length < 1e-8:
                continue

            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=cyl_radius,
                cone_radius=cone_radius,
                cylinder_height=0.8 * length,
                cone_height=0.2 * length
            )
            arrow.compute_vertex_normals()
            arrow.paint_uniform_color(colors[i])

            # Rotate from z-axis to vector
            direction = vec / length
            z_axis = np.array([0, 0, 1])
            v = np.cross(z_axis, direction)
            c = np.dot(z_axis, direction)
            if np.linalg.norm(v) < 1e-8:
                R = np.eye(3) if c > 0 else -np.eye(3)
            else:
                vx = np.array([[0, -v[2], v[1]],
                               [v[2], 0, -v[0]],
                               [-v[1], v[0], 0]])
                R = np.eye(3) + vx + vx @ vx * (1 / (1 + c))

            arrow.rotate(R, center=np.zeros(3))
            arrow.translate(p0)
            arrows.append(arrow)
        return arrows

    def update(self, vectors):
        # Remove old arrows
        for a in self.arrows:
            self.vis.remove_geometry(a, reset_bounding_box=False)

        # Create new arrows
        self.arrows = self._create_arrows(vectors)
        for a in self.arrows:
            self.vis.add_geometry(a, reset_bounding_box=False)

        # Refresh view
        self.vis.poll_events()
        self.vis.update_renderer()

    def run_once(self):
        """Keeps the window open until user closes it manually."""
        self.vis.run()
        self.vis.destroy_window()
