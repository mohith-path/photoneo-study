import numpy as np
import caml_cell as cell
import caml_core as core

from utils import get_overlap_metrics
from scanners import Phoxi, MotioncamOld

CAMERA = "phoxi"
VISUALIZE = False

PARTS = [("Fabtech", "shelf_bracket"), ("Fabtech", "side_left"), ("Fabtech", "base_bracket"), ("Hutchens", "left_wall")]


def evaluate():
    environment = cell.environments.Environment()
    scanner = Phoxi.get_scanner()
    environment.broadcast_transform(core.geometry.Pose(parent_frame="world", child_frame="scanner"))

    metrics = []

    for i, (CUSTOMER, PART) in enumerate(PARTS):
        for j in range(2):
            # Load data
            if CUSTOMER == "Fabtech":
                part_template_path = f"/home/mohith/oasis/CAMLPartTemplates/{CUSTOMER}/Organizer/{PART}"
            elif CUSTOMER == "Hutchens":
                part_template_path = f"/home/path/projects/oasis/CAMLPartTemplates/{CUSTOMER}/20732/{PART}"
            part_template = core.io.load_part_template(part_template_path)

            if CAMERA == "phoxi":
                scan_path = f"data/{CAMERA}/snapshot_{2 * i + j:05d}/observations/left_sensor_holder_eye_1_camera_optical/"
                scan = core.io.load_point_cloud(scan_path + "point_cloud.ply")

            # Generate ray-casting-scene
            camera_T_part = core.io.load_pose(scan_path + "camera_T_part.yaml")  # This pose was acquired through registration
            ray_cast_scene = part_template.get_mesh()
            ray_cast_scene.transform(camera_T_part)

            # Load Camera
            simulated_scan = scanner.ray_cast(
                mesh=ray_cast_scene,
                reference_T_mount=core.geometry.Pose(),
                individual_filters=[
                    core.ray_casting.filters.AngleOfIncidence(1.2),
                ],
                combined_filters=[
                    # core.ray_casting.filters.HalfwayAngle(
                    #     min_angle=np.deg2rad(10),
                    #     max_angle=np.deg2rad(60),
                    # )
                ],
            )

            simulated_scan = simulated_scan.get_observations()[0].get_simulated_point_cloud()

            metrics.append(get_overlap_metrics(simulated_scan, scan, VISUALIZE))

    metrics = np.array(metrics)

    print("Metrics: ", metrics.mean(axis=0))
    print("\nValues: \n", metrics)


if __name__ == "__main__":
    evaluate()
