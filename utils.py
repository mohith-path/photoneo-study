from typing import Tuple

import numpy as np
import caml_core as core
import caml_algorithms as algorithms


def get_camera_T_part(
    scan: core.geometry.PointCloud, part_template: core.data_structures.PartTemplate, observation_path: str = None
) -> core.geometry.Pose:
    """
    Register part to t a scan

    :param scan: Scan from the sensor in the sensor's frame
    :param part_template: Part template
    :param observation_path: Path of the observation containing the scan
    :return: Part pose in camera frame (camera_T_part)
    """
    icp = algorithms.registration.UserSelectedCorrespondences()

    target = scan
    source = part_template.get_mesh().sample_points_uniformly(500000)

    target_T_source = icp.register(
        source=source,
        target=target,
    )

    new_pc = source.clone()
    new_pc.transform(target_T_source)
    core.visualization.draw_geometries_with_hiding([], [new_pc, target])

    reregister = input("Register again (y/n): ")
    if reregister == "y":
        mesh = part_template.get_mesh()
        mesh.transform(target_T_source)

        image_dimensions = (1080, 1440)
        intrinsics = np.array(
            [[1721.75, 0, ((image_dimensions[1] - 1) / 2) + 1], [0, 1721.3, ((image_dimensions[0] - 1) / 2) + 1], [0, 0, 1]],
        )

        simulated_target = core.ray_casting.ray_cast_from_camera(
            mesh=mesh, extrinsics=core.geometry.Pose(), intrinsics=intrinsics, image_dimensions=image_dimensions
        )

        icp = algorithms.registration.IterativeICP()
        target_T_simulated_target = icp.register(
            source=simulated_target, target=target, initial_transform=core.geometry.Pose(), number_of_iterations=2
        )
        simulated_target.transform(target_T_simulated_target)
        core.visualization.draw_geometries_with_hiding([simulated_target, target])

        target_T_source = target_T_simulated_target * target_T_source
        new_pc.transform(target_T_simulated_target)

    cropped_point_cloud = scan.intersect(other=new_pc, radius=0.006)
    cropped_point_cloud = scan

    core.visualization.draw_geometries_with_hiding([], [scan, cropped_point_cloud])

    if observation_path is not None:
        save_data = input("Save data (y/n): ")
        if save_data == "y":
            # TODO:
            # core.io.save_point_cloud(scan_path + "point_cloud.ply", cropped_point_cloud)
            core.io.save_pose(observation_path + "camera_T_part.yaml", target_T_source)

    return target_T_source, cropped_point_cloud


def get_overlap_metrics(prediction: core.geometry.PointCloud, target: core.geometry.PointCloud, visualize: bool) -> Tuple[float, float]:
    """
    Compute metrics to determine how good the camera config is for simulating scans
    Measures overlap between scanned and real point cloud

    :param prediction: Simulated point cloud being evaluated
    :param target: Real scan being compared against
    :param visualize: Flag to visualize the overlap
    :return: The overlap ratio of the simulated point cloud with the scan and the overlap of the scan with the simulated point cloud
    """
    prediction_num_pts = len(prediction)
    target_num_pts = len(target)

    prediction_overlap = prediction.intersect(other=target, radius=0.002)
    prediction_num_overlap_pts = len(prediction_overlap)

    target_overlap = target.intersect(other=prediction, radius=0.002)
    target_num_overlap_pts = len(target_overlap)

    if visualize:
        core.visualization.draw_geometries_with_hiding([], [prediction, prediction_overlap, target, target_overlap])

    return (prediction_num_overlap_pts / prediction_num_pts, target_num_overlap_pts / target_num_pts)
