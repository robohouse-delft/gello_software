from multiprocessing import Process

import toml

from gello.cameras.realsense_camera import RealSenseCamera, get_device_ids
from gello.zmq_core.camera_node import ZMQServerCamera

def launch_server(port: int, camera_id: str, config):
    camera = RealSenseCamera(camera_id)
    server = ZMQServerCamera(camera, port=port, host=config["hostname"])
    print(f"Starting camera server on port {port}")
    server.serve()


def main(config):
    ids = get_device_ids()
    camera_port = 5000
    camera_servers = []
    for camera_id in ids:
        # start a python process for each camera
        print(f"Launching camera {camera_id} on port {camera_port}")
        camera_servers.append(
            Process(target=launch_server, args=(camera_port, camera_id, config))
        )
        camera_port += 1

    for server in camera_servers:
        server.start()


if __name__ == "__main__":
    config = toml.load("./config.toml")
    main(config["camera_server"])
