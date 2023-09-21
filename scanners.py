import numpy as np
import caml_cell as cell
import caml_core as core


class Phoxi:
    intrinsics = np.array([[2362.5871586302883, 0.0, 959.453584632744], [0.0, 2363.576450360493, 759.2781971422867], [0.0, 0.0, 1.0]])

    distortion_coefficients = np.array(
        [
            -0.24146227495473033,
            0.18577673476723738,
            -0.0005776280396240931,
            -0.0006494314999141272,
            -0.11864417907836954,
        ]
    )
    distortion_coefficients = np.zeros(5)

    image_dimension = (1544, 2064)

    scanner_T_camera = core.geometry.Pose(translation=(-0.175, 0, 0), euler_angles=(0, 0.2234, 0))
    scanner_T_projector = core.geometry.Pose(translation=(0.175, 0, 0), euler_angles=(0, -0.2234, 0))

    camera_T_scanner = scanner_T_camera.inverse() * scanner_T_projector

    @staticmethod
    def get_scanner():
        camera = cell.sensors.cameras.Camera(
            frame_name="camera",
            exposure=100,
            intrinsics=Phoxi.intrinsics,
            distortion_coefficients=Phoxi.distortion_coefficients,
            image_dimensions=Phoxi.image_dimension,
            mount_frame="world",
        )

        projector = cell.sensors.projectors.Projector(
            frame_name="projector", fan_angle_x=0.8, fan_angle_y=0.8, number_of_rays=2048**2, mount_frame="camera"
        )

        scanner = cell.sensors.scanners.Scanner(
            frame_name="scanner", projector=projector, receivers=[camera], scanning_range=(0, 2), mount_frame="world"
        )

        scanner.set_mount_T_sensor(core.geometry.Pose())
        scanner.receivers[0].set_mount_T_sensor(core.geometry.Pose())
        scanner.projector.set_mount_T_sensor(Phoxi.camera_T_scanner)

        return scanner


class MotioncamOld:
    intrinsics = np.array([[1721.7503573972544, 0.0, 814.8148164226212], [0.0, 1721.2981310810762, 605.0013672281955], [0.0, 0.0, 1.0]])

    distortion_coefficients = np.zeros(5)

    image_dimension = (1200, 1680)

    camera_T_scanner = core.geometry.Pose(translation=[0.2, 0, 0], euler_angles=[0, -0.3, 0])

    @staticmethod
    def get_scanner():
        camera = cell.sensors.cameras.Camera(
            frame_name="camera",
            exposure=100,
            intrinsics=MotioncamOld.intrinsics,
            distortion_coefficients=MotioncamOld.distortion_coefficients,
            image_dimensions=MotioncamOld.image_dimension,
            mount_frame="world",
        )

        projector = cell.sensors.projectors.Projector(
            frame_name="projector", fan_angle_x=0.8, fan_angle_y=0.8, number_of_rays=2048**2, mount_frame="camera"
        )

        scanner = cell.sensors.scanners.Scanner(
            frame_name="scanner", projector=projector, receivers=[camera], scanning_range=(0, 2), mount_frame="world"
        )

        scanner.set_mount_T_sensor(core.geometry.Pose())
        scanner.receivers[0].set_mount_T_sensor(core.geometry.Pose())
        scanner.projector.set_mount_T_sensor(MotioncamOld.camera_T_scanner)

        return scanner
