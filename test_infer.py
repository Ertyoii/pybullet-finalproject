from infer import *
from grasp_robot import *


def test_infer(view_matrix, projection_matrix):
    record = "./NaiveNet200.pth"
    seg_img, obj, obj_pos, obj_rot = generate_random_object(view_matrix, projection_matrix)
    res = infer(seg_img, record)

    robot = build_kuka()

    x = obj_pos[0] - robot.getObservation()[0]
    y = obj_pos[1] - robot.getObservation()[1]
    a = math.pi / 2 - obj_rot

    print("target: {} {} {}".format(x, y, a))
    print("actual: {} {} {}".format(res[0], res[1], res[2]))
    adjust_and_down(robot, res[0], res[1], res[2])

    return grasp(robot, obj)


if __name__ == "__main__":
    view_matrix, projection_matrix = init(0)
    for i in range(100):
        if not test_infer(view_matrix, projection_matrix):
            print("Failed.")
        else:
            print("Pass:", i)
    p.disconnect()
