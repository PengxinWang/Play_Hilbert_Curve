import os
import torch
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from hilbert.dataset import ModelNetDataset
from hilbert.transform import GridSample
from hilbert.default import encode

def serialization(data, order=["hilbert"], depth=None, shuffle_orders=False):
    """
    Point Cloud Serialization

    relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
    """
    if depth is None:
        depth = int(data["grid_coord"].max()).bit_length()
    data["serialized_depth"] = depth
    assert depth * 3 <= 63
    assert depth <= 16
    code = [
        encode(data["grid_coord"], depth=depth, order=order_) for order_ in order
    ]
    code = torch.stack(code)
    order = torch.argsort(code)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(0, code.shape[1], device=order.device).repeat(
            code.shape[0], 1
            ),
        )

    if shuffle_orders:
        perm = torch.randperm(code.shape[0])
        code = code[perm]
        order = order[perm]
        inverse = inverse[perm]

    data["serialized_code"] = code           
    data["serialized_order"] = order         
    data["serialized_inverse"] = inverse 
    return data  

def build_label_to_color(labels):
    cmap = plt.get_cmap('plasma')
    num_labels = len(labels)
    normalized_indices = np.linspace(0,1,num_labels)
    colors = [cmap(i)[:3] for i in normalized_indices]
    label_to_color = {label: list(color) for label,color in zip(labels,colors)}
    return label_to_color

def plot_pcd(pcd, labels, save_path, front_vector=[0.2, 0, 0.8], up_vector=[0., 1., 0.], zoom_factor=0.8, 
             plot_mesh=True, radii=[1.5, 2.0, 6.0]):
    points = pcd["coord"]
    print(f'num of points: {points.shape[0]}')
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # apply color to points
    label_to_color = build_label_to_color(labels)
    colors = np.array([label_to_color[label] for label in labels])
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis=o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    if plot_mesh:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=50)
        print("Performing Ball-Pivoting Algorithm(BPA)")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        print(f'mean distance: {avg_dist}')
        radii = [avg_dist*r for r in radii]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        vis.add_geometry(mesh)
    else:
        vis.add_geometry(pcd)

    # set camera position
    ctr = vis.get_view_control()
    ctr.set_front(front_vector)
    ctr.set_up(up_vector)
    ctr.set_zoom(zoom_factor)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)
    vis.destroy_window()
    print(f'Saved image to: {save_path}')

def main(args):
    data_root = args.data_root
    idx = args.idx
    patch_size = args.patch_size

    dataset = ModelNetDataset(data_root=data_root)
    print(f'size of dataset: {len(dataset)}')
    data = dataset.get_data(idx=idx)
    grid_sample = GridSample(keys=["coord", "normal"], grid_size=0.01, return_grid_coord=True)
    data = grid_sample(data)
    data = serialization(data=data)
    indexes = data["serialized_order"]
    n_points = indexes.numel()
    labels = []
    num_full_patchs = int(n_points/patch_size)
    for i in range(0, num_full_patchs):
        labels.extend([i]*patch_size)
    labels.extend([num_full_patchs+1]*(len(data["coord"])-len(labels)))

    if args.plot_mesh:
        save_path = os.path.join(args.save_dir, f'{data["name"]}_mesh.png')
    else:
        save_path = os.path.join(args.save_dir, f'{data["name"]}.png')

    plot_pcd(pcd=data, 
             labels=labels, 
             save_path = save_path,
             plot_mesh = args.plot_mesh)   

if __name__=="__main__":
    parser=argparse.ArgumentParser('visualization')
    parser.add_argument('--data_root', default="D:\PointNet\PointNet\data\modelnet40_normal_resampled")
    parser.add_argument('--save_dir', default="imgs")
    parser.add_argument('--idx', default=8000, type=int)
    parser.add_argument('--patch_size', default=64, type=int)
    parser.add_argument('--plot_mesh', action='store_true')
    args = parser.parse_args()
    main(args)