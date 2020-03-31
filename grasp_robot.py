import math
import time
import random

import numpy as np
import pybullet as p
import pybullet_data as pdata
from pybullet_envs.bullet import kuka


def init(useGUI):
    if useGUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    p.setAdditionalSearchPath(pdata.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # Compute the view and image from camera
    view_matrix = p.computeViewMatrix([0.575, 0, 0.5], [0.575, 0, 0], [-1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1,
        nearVal=0.1,
        farVal=0.55)

    return view_matrix, projection_matrix


def get_object_height(uid):
    pos, orn = p.getBasePositionAndOrientation(uid)
    return pos[2]


def grasp(robot, obj):
    robot.applyAction([0, 0, 0, 0, 0])

    start = time.time()
    # Wait until the gripper grasps the object.
    while p.getContactPoints(obj, robot.kukaUid, -1, 10) == ():
        if time.time() - start > 5:
            return False
        p.stepSimulation()

    robot.applyAction([0, 0, 0.2, 0, 0])

    for _ in range(200):
        if get_object_height(obj) > 0.2:
            return True
        p.stepSimulation()

    return False


def build_and_grasp(view_matrix, projection_matrix):
    seg_img, obj, obj_pos, obj_rot = generate_random_object(view_matrix, projection_matrix)

    robot = build_kuka()

    x = obj_pos[0] - robot.getObservation()[0]
    y = obj_pos[1] - robot.getObservation()[1]
    a = math.pi / 2 - obj_rot

    adjust_and_down(robot, x, y, a)

    return grasp(robot, obj), seg_img, [x, y, a]


def adjust_and_down(robot, x, y, a):
    robot.applyAction([x, y, -0.22, a, 0.3])

    # Wait until the gripper is at surface level.
    while robot.getObservation()[2] > 0.24:
        p.stepSimulation()

    # Wait until the gripper stops moving.
    while p.getLinkState(robot.kukaUid, 13, 1)[6][0] > 1e-5:
        p.stepSimulation()


def generate_random_object(view_matrix, projection_matrix):
    p.resetSimulation()
    hw = 128
    # Load object
    p.loadURDF("plane.urdf")
    # x_bound: [0.5, 0.65]
    # y_bound: [-0.17, 0.21]
    obj_x = random.uniform(0.55, 0.6)
    obj_y = random.uniform(-0.1, 0.15)

    obj_pos = [obj_x, obj_y, 0.02]
    obj_rot = random.uniform(-math.pi / 3, math.pi / 3)
    obj_euler_orient = [math.pi / 2, 0, obj_rot]
    obj_quater_orient = p.getQuaternionFromEuler(obj_euler_orient)
    obj = p.loadURDF("./0002/0002.urdf", obj_pos, obj_quater_orient)

    # Get image of the objects without loading the robot
    _, _, _, _, seg_img = p.getCameraImage(width=hw,
                                           height=hw,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=projection_matrix)

    seg_img = np.reshape(seg_img, (hw, hw, 1))
    return seg_img, obj, obj_pos, obj_rot


def build_kuka():
    # Load Kuka robot.
    robot = kuka.Kuka(urdfRootPath=pdata.getDataPath())

    # Remove the tray object.
    p.removeBody(robot.trayUid)
    return robot


if __name__ == "__main__":
    view_matrix, projection_matrix = init(1)
    while 1:
        success, seg_img, [x, y, a] = build_and_grasp(view_matrix, projection_matrix)
