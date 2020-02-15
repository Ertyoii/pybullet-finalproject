import math
import time
import random

import cv2
import csv
import numpy as np
import pybullet as p
import pybullet_data as pdata
from pybullet_envs.bullet import kuka
from infer import *


def init(useGUI):
    if (useGUI):
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
    view_matrix = p.computeViewMatrix([0.575, 0, 0.7], [0.6, 0, 0], [-1, 0, 0])
    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1,
        nearVal=0.1,
        farVal=1.6)

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


def test(view_matrix, projection_matrix):
    seg_img, obj, obj_pos, obj_rot = generate_random_object(view_matrix, projection_matrix)

    # Load Kuka robot.
    robot = kuka.Kuka(urdfRootPath=pdata.getDataPath(), timeStep=1. / 240.)

    # Remove the tray object.
    p.removeBody(robot.trayUid)

    x = obj_pos[0] - robot.getObservation()[0]
    y = obj_pos[1] - robot.getObservation()[1]
    a = math.pi / 2 - obj_rot

    generate_robot_movement(robot, x, y, a)

    return grasp(robot, obj), seg_img, [x, y, a]


def generate_robot_movement(robot, x, y, a):
    robot.applyAction([x, y, -0.22, a, 0.3])

    # Wait until the gripper is at surface level.
    while robot.getObservation()[2] > 0.24:
        p.stepSimulation()

    # Wait until the gripper stops moving.
    while p.getLinkState(robot.kukaUid, 13, 1)[6][0] > 1e-5:
        p.stepSimulation()


def generate_random_object(view_matrix, projection_matrix):
    p.resetSimulation()

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
    _, _, _, d, seg_img = p.getCameraImage(width=256,
                                           height=256,
                                           viewMatrix=view_matrix,
                                           projectionMatrix=projection_matrix)

    seg_img = np.reshape(seg_img, (256, 256, 1))
    return seg_img, obj, obj_pos, obj_rot


def test_infer(view_matrix, projection_matrix):
    record = "./NaiveNet200.pth"
    seg_img, obj, obj_pos, obj_rot = generate_random_object(view_matrix, projection_matrix)
    res = infer(seg_img, record)

    # Load Kuka robot.
    robot = kuka.Kuka(urdfRootPath=pdata.getDataPath(), timeStep=1. / 240.)

    # Remove the tray object.
    p.removeBody(robot.trayUid)

    x = obj_pos[0] - robot.getObservation()[0]
    y = obj_pos[1] - robot.getObservation()[1]
    a = math.pi / 2 - obj_rot

    print("target: {} {} {}".format(x, y, a))
    print("actual: {} {} {}".format(res[0], res[1], res[2]))
    generate_robot_movement(robot, res[0], res[1], res[2])

    return grasp(robot, obj)


def build_dataset(view_matrix, projection_matrix, n):
    count = 0
    f = open("label.csv", "a")
    writer = csv.writer(f)
    while count < n:

        success, seg_img, xya = test(view_matrix, projection_matrix)
        if success:
            print(count)
            cv2.imwrite("./data/"+str(count)+".jpeg", seg_img)
            writer.writerow(xya)
            count += 1


if __name__ == "__main__":
    view_matrix, projection_matrix = init(0)
    # for i in range(10000):
    #     if not test(view_matrix, projection_matrix)[0]:
    #         print("Failed.")
    #     else:
    #         print("Pass:", i)
    for i in range(100):
        if not test_infer(view_matrix, projection_matrix):
            print("Failed.")
        else:
            print("Pass:", i)
    p.disconnect()
