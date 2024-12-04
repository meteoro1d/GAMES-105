import time
from typing import TextIO, List
import numpy as np
from scipy.spatial.transform import Rotation as R


def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i + 1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1, -1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    file = open(bvh_file_path, 'r')
    for line in file:
        strs = line.split()
        if strs[0] == 'ROOT':
            parse_hierarchy(strs[1], -1, file, joint_name, joint_parent, joint_offset)
    file.close()

    joint_offset = np.array(joint_offset).reshape(-1, 3)
    return joint_name, joint_parent, joint_offset


def parse_hierarchy(name: str, parent: int, file: TextIO, joint_name: List[str], joint_parent: List[int],
                    joint_offset: List[float]):
    joint_name.append(name)
    joint_parent.append(parent)
    joint_index = len(joint_name) - 1

    for line in file:
        line = line.strip()
        strs = line.split()

        if strs[0] == 'OFFSET':
            joint_offset.append(float(strs[1]))
            joint_offset.append(float(strs[2]))
            joint_offset.append(float(strs[3]))
        elif strs[0] == 'JOINT':
            parse_hierarchy(strs[1], joint_index, file, joint_name, joint_parent, joint_offset)
        elif strs[0] == 'End':
            parse_hierarchy(f'{name}_end', joint_index, file, joint_name, joint_parent, joint_offset)
        elif strs[0] == '}':
            return


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    joint_positions = None
    joint_orientations = None

    frame_motion_data = motion_data[frame_id]
    j = 0
    for i in range(0, len(joint_name)):
        if joint_name[i] == "RootJoint":
            # position
            position = np.array([frame_motion_data[j], frame_motion_data[j + 1], frame_motion_data[j + 2]]).reshape(-1,
                                                                                                                    3)
            j += 3
            joint_positions = position

            # orientation
            orientation = R.from_euler('XYZ',
                                       [frame_motion_data[j], frame_motion_data[j + 1], frame_motion_data[j + 2]],
                                       degrees=True).as_quat().reshape(-1, 4)
            j += 3
            joint_orientations = orientation
        else:
            parent = joint_parent[i]
            offset = joint_offset[i]
            parent_orientation = R.from_quat(joint_orientations[parent])

            # position
            position = (parent_orientation.apply(offset) + joint_positions[parent]).reshape(-1, 3)
            joint_positions = np.concatenate((joint_positions, position))

            # orientation
            if '_end' in joint_name[i]:
                joint_orientations = np.concatenate((joint_orientations, np.zeros([1, 4])))
            else:
                local_orientation = R.from_euler('XYZ', [frame_motion_data[j], frame_motion_data[j + 1],
                                                         frame_motion_data[j + 2]], degrees=True)
                j += 3
                orientation = (parent_orientation * local_orientation).as_quat().reshape(-1, 4)
                joint_orientations = np.concatenate((joint_orientations, orientation))

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data_a = load_motion_data(A_pose_bvh_path)
    motion_data = np.zeros(motion_data_a.shape)

    joint_name_t, _, _ = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a, _, _ = part1_calculate_T_pose(A_pose_bvh_path)

    joint_name_t = [item for item in joint_name_t if not item.endswith('_end')]
    joint_name_a = [item for item in joint_name_a if not item.endswith('_end')]

    for frame_id in range(0, motion_data_a.shape[0]):
        for joint_index in range(0, len(joint_name_t)):
            joint_name = joint_name_t[joint_index]

            # 根节点位置
            if joint_name == 'RootJoint':
                motion_data[frame_id][0] = motion_data_a[frame_id][0]
                motion_data[frame_id][1] = motion_data_a[frame_id][1]
                motion_data[frame_id][2] = motion_data_a[frame_id][2]

            # 获得的A-pose在joint上的旋转
            joint_index_in_a = joint_name_a.index(joint_name)
            euler_index = 3 + joint_index_in_a * 3
            r_a_euler = motion_data_a[frame_id][euler_index: euler_index + 3]

            if joint_name == 'lShoulder':
                r_t = R.from_euler("XYZ", r_a_euler, degrees=True) * R.from_euler("XYZ", [0, 0, -45], degrees=True)
                r_t_euler = r_t.as_euler("XYZ", degrees=True)
            elif joint_name == 'rShoulder':
                r_t = R.from_euler("XYZ", r_a_euler, degrees=True) * R.from_euler("XYZ", [0, 0, 45], degrees=True)
                r_t_euler = r_t.as_euler("XYZ", degrees=True)
            else:
                r_t_euler = r_a_euler

            motion_data[frame_id][3 + joint_index * 3] = r_t_euler[0]
            motion_data[frame_id][3 + joint_index * 3 + 1] = r_t_euler[1]
            motion_data[frame_id][3 + joint_index * 3 + 2] = r_t_euler[2]

    return motion_data
