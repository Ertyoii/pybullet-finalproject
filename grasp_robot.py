import cv2
import math
import random
import numpy as np

import pybullet as p
import pybullet_data as pdata
from pybullet_envs.bullet import kuka

"""
    TODO:
    4. Build a naive model to infer x, y, a from images

    DONE:
    1. Find the center of object
    2. Figure out the x, y input's relationship with center
    3. Generate images and corresponding x, y, a
"""


def init():
    p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    p.setAdditionalSearchPath(pdata.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)


def get_object_height(uid):
    pos, orn = p.getBasePositionAndOrientation(uid)
    return pos[2]


def grasp(robot, obj):
    robot.applyAction([0, 0, 0, 0, 0])

    # Wait until the gripper grasps the object.
    while p.getContactPoints(obj, robot.kukaUid, -1, 10) == ():
        p.stepSimulation()

    robot.applyAction([0, 0, 0.2, 0, 0])

    for _ in range(1000):
        if get_object_height(obj) > 0.2:
            return True
        p.stepSimulation()

    return False


def test():
    p.resetSimulation()

    # Compute the view and image from camera
    view_matrix = p.computeViewMatrix([0.575, 0, 0.7], [0.6, 0, 0], [-1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1,
        nearVal=0.1,
        farVal=1.6)

    # Load plane and object
    p.loadURDF("plane.urdf")

    # x_bound: [0.5, 0.65]
    # y_bound: [-0.17, 0.21]
    x = random.uniform(0.55, 0.6)
    y = random.uniform(-0.1, 0.15)

    obj_pos = [x, y, 0.02]
    obj_rot = random.uniform(-math.pi / 3, math.pi / 3)
    obj_euler_orient = [math.pi / 2, 0, obj_rot]
    obj_quater_orient = p.getQuaternionFromEuler(obj_euler_orient)
    obj = p.loadURDF("0002.urdf", obj_pos, obj_quater_orient)

    # Get image of the objects without loading the robot
    _, _, _, _, seg_img = p.getCameraImage(width=256,
                                           height=256,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=projection_matrix)

    seg_img = np.reshape(seg_img, (256, 256, 1))
    seg_img[seg_img == 1] = 124
    cv2.imwrite("hi.jpeg", seg_img)

    # Load Kuka robot.
    robot = kuka.Kuka(urdfRootPath=pdata.getDataPath(), timeStep=1. / 240.)

    # Remove the tray object.
    p.removeBody(robot.trayUid)

    x = obj_pos[0] - robot.getObservation()[0]
    y = obj_pos[1] - robot.getObservation()[1]
    a = math.pi / 2 - obj_rot
    robot.applyAction([x, y, -0.22, a, 0.3])

    # Wait until the gripper is at surface level.
    while robot.getObservation()[2] > 0.24:
        p.stepSimulation()

    # Wait until the gripper stops moving.
    while p.getLinkState(robot.kukaUid, 13, 1)[6][0] > 1e-5:
        p.stepSimulation()

    return grasp(robot, obj)


if __name__ == "__main__":
    init()
    # for i in range(10000):
    #     if not test():
    #         print("Failed.")
    #     else:
    #         print("Pass:", i)
    test()
    p.disconnect()
