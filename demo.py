import math
import pybullet as p
import pybullet_data as pdata
from pybullet_envs.bullet import kuka

if __name__ == "__main__":
    p.connect(p.GUI)

    p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    p.setAdditionalSearchPath(pdata.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[0, 0, 0.5],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[-1, 0, 0])

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=45.0,
        aspect=1,
        nearVal=0.1,
        farVal=0.5)

    p.loadURDF("plane.urdf")
    # obj_quater_orient = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
    # obj = p.loadURDF("cube_small.urdf", [0, 0, 0], obj_quater_orient)

    p.getCameraImage(width=256,
                     height=256,
                     viewMatrix=view_matrix,
                     projectionMatrix=projection_matrix)

    robot = kuka.Kuka(urdfRootPath=pdata.getDataPath())
    p.removeBody(robot.trayUid)

    # robot.applyAction([0, 0, -0.22, math.pi/2, 0])

    while 1:
        p.stepSimulation()
