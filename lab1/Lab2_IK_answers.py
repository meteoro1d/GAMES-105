import numpy as np
from scipy.spatial.transform import Rotation as R
from lab1.task2_inverse_kinematics import MetaData
import torch

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
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # joint offset
    path_joint_offset = torch.zeros((len(path), 3))

    print(meta_data.root_joint)

    for i in range (len(path)):
        if i == 0:
            parent_position = meta_data.joint_initial_position[path[0]]
        else:
            parent_position = meta_data.joint_initial_position[path[i - 1]]

        path_joint_offset[i] = torch.from_numpy(meta_data.joint_initial_position[path[i]] - parent_position)

    print(path_joint_offset)

    #
    theta = torch.zeros((len(path), 3), requires_grad=True)

    iteration = 10

    for it in range(iteration):
        last_joint_orientation = None
        last_joint_position = None

        for i in range(len(path)):
            if i == 0:
                last_joint_position = meta_data.joint_initial_position[path[0]]
                last_joint_orientation = R.from_euler('XYZ', theta[i].numpy(), degrees=True)


    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations