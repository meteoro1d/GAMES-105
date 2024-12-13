import numpy as np
from scipy.spatial.transform import Rotation as R
from lab1.task2_inverse_kinematics import MetaData

MAX_ITERATIONS = 1000
LEARNING_RATE = 0.1

def part1_inverse_kinematics(meta_data: MetaData, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, _, _, _ = meta_data.get_path_from_root_to_end()
    path_joints_positions, path_joints_orientations = get_path_joints_initial_positions_and_orientations(path, joint_positions, joint_orientations)
    theta = get_initial_theta(path_joints_orientations, R.identity())

    for it in range(1, MAX_ITERATIONS + 1):
        theta = update_theta(theta, path, path_joints_positions, path_joints_orientations, target_pose)
        update_path_joint_positions_and_orientations(meta_data, theta, path, path_joints_positions, path_joints_orientations, R.identity())

        distance = np.linalg.norm(path_joints_positions[-1] - target_pose)
        print(f'iteration {it}: target_position = {target_pose}, end_position = {path_joints_positions[-1]}, distance = {distance}')

        # 到达目标或最大次数，跳出
        if distance <= 0.01:
            break

    # 计算所有节点最终的位置和朝向
    for i in range(len(path)):
        joint_positions[path[i]] = path_joints_positions[i]
        joint_orientations[path[i]] = path_joints_orientations[i].as_quat()

    for joint in range(len(meta_data.joint_name)):
        if joint in path:
            continue
        else:
            parent = meta_data.joint_parent[joint]
            joint_orientations[joint] = joint_orientations[parent]
            joint_positions[joint] = joint_positions[parent] + R.from_quat(joint_orientations[parent]).apply(get_joint_offset(joint, parent, meta_data))

    return joint_positions, joint_orientations


# 得到从root节点到end节点的路径的初始位置和朝向
def get_path_joints_initial_positions_and_orientations(path, joint_positions, joint_orientations):
    path_joints_positions = np.zeros((len(path), 3))
    path_joints_orientations = []

    for i in range(len(path)):
        path_joints_positions[i] = joint_positions[path[i]]
        path_joints_orientations.append(R.from_quat(joint_orientations[path[i]]))

    return path_joints_positions, path_joints_orientations


# 初始theta
def get_initial_theta(path_joints_orientations, root_parent_orientation):
    theta = np.zeros((len(path_joints_orientations), 3))
    for i in range(len(path_joints_orientations)):
        if i == 0:
            theta[i] = (root_parent_orientation.inv() * path_joints_orientations[i]).as_euler('XYZ')
        else:
            theta[i] = (path_joints_orientations[i - 1].inv() * path_joints_orientations[i]).as_euler('XYZ')
    return theta


# 更新参数
def update_theta(theta, path, path_joints_positions, path_joints_orientations, target_pose, update_root=False):
    end_joint_position = path_joints_positions[-1]
    jacobian_T = np.zeros((len(path) * 3, 3))

    if update_root:
        start = 0
    else:
        start = 1

    for i in range(start, len(path)):
        a = end_joint_position - path_joints_positions[i]
        rotation_matrix = path_joints_orientations[i].as_matrix()

        for axis in range(3):
            jacobian_T[i * 3 + axis] = np.cross(a, rotation_matrix[:, axis])

    return theta - LEARNING_RATE * np.matmul(jacobian_T, target_pose - end_joint_position).reshape(theta.shape)


# 根据参数更新位置
def update_path_joint_positions_and_orientations(meta_data, theta, path, path_joints_positions, path_joints_orientations, root_parent_orientation):
    for i in range(len(path)):
        if i == 0:
            path_joints_orientations[i] = root_parent_orientation * R.from_euler('XYZ', theta[i])
        else:
            path_joints_orientations[i] = path_joints_orientations[i - 1] * R.from_euler('XYZ', theta[i])
            path_joints_positions[i] = path_joints_positions[i - 1] + path_joints_orientations[i - 1].apply(get_joint_offset(path[i], path[i - 1], meta_data))


def get_joint_offset(joint, parent, meta_data):
    return meta_data.joint_initial_position[joint] - meta_data.joint_initial_position[parent]


def part2_inverse_kinematics(meta_data: MetaData, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    path, path_name, _, _ = meta_data.get_path_from_root_to_end()

    # path上root joint的parent joint的朝向
    root_parent = meta_data.joint_parent[path[0]]
    root_parent_rotation = R.from_quat(joint_orientations[root_parent])

    path_joints_positions, path_joints_orientations = get_path_joints_initial_positions_and_orientations(path, joint_positions, joint_orientations)
    theta = get_initial_theta(path_joints_orientations, root_parent_rotation)

    # 目标位置
    target_pose = np.array([joint_positions[0][0] + relative_x, target_height, joint_positions[0][2] + relative_z])

    for it in range(1, MAX_ITERATIONS + 1):
        theta = update_theta(theta, path, path_joints_positions, path_joints_orientations, target_pose, True)
        update_path_joint_positions_and_orientations(meta_data, theta, path, path_joints_positions, path_joints_orientations, root_parent_rotation)

        distance = np.linalg.norm(path_joints_positions[-1] - target_pose)
        print(f'iteration {it}: target_position = {target_pose}, end_position = {path_joints_positions[-1]}, distance = {distance}')

        # 到达目标或最大次数，跳出
        if distance <= 0.01:
            break

    # apply ik result
    for i in range(len(path)):
        joint_positions[path[i]] = path_joints_positions[i]
        joint_orientations[path[i]] = path_joints_orientations[i].as_quat()
    return joint_positions, joint_orientations


def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """

    return joint_positions, joint_orientations
