import toml
import numpy as np
from typing import List, Dict

from gello.zmq_core.camera_node import ZMQClientCamera

def main(camera_clients: List[Dict], server_hostname):
    cameras = []
    import cv2

    images_display_names = []
    for camera_client in camera_clients:
        port = camera_client["port"]
        cameras.append(ZMQClientCamera(port=port, host=server_hostname))
        images_display_names.append(f"image_{port}")
        cv2.namedWindow(images_display_names[-1], cv2.WINDOW_NORMAL)

    while True:
        for display_name, camera in zip(images_display_names, cameras):
            image, depth = camera.read()
            stacked_depth = np.dstack([depth, depth, depth]).astype(np.uint8)
            image_depth = cv2.hconcat([image[:, :, ::-1], stacked_depth])
            cv2.imshow(display_name, image_depth)
            cv2.waitKey(1)


if __name__ == "__main__":
    config = toml.load("./config.toml")
    main(config["camera_clients"], config["camera_server"]["hostname"])
