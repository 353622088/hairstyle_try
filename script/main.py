# coding:utf-8 
'''
created on 2018/9/27

@author:Dxq
'''
from PIL import Image
import os
import scipy.io as scio
import numpy as np
import cv2
import functools
import time
from common.config import DESKTOP as Desktop
from common.utils import get_baseInfo_tx


def time_cal(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        t1 = time.time()
        r = func(*args, **kw)  # 先让函数运行一次,防止直接输出，将其赋值给一个变量
        if time.time() - t1 > 0.001:
            print('函数%s执行的时间为：%f' % (func.__name__, time.time() - t1))
        return r

    return wrapper


@time_cal
def get_landmark_dict(file_path):
    mat_file = file_path.split(".")[0] + '.mat'
    if os.path.exists(mat_file):
        landmark_dict = scio.loadmat(mat_file)
    else:
        landmark_dict = get_baseInfo_tx(file_path)
        if landmark_dict['roll'] != 0:
            Image.open(file_path).rotate(-landmark_dict['roll']).save(file_path)
            landmark_dict = get_baseInfo_tx(file_path)
        scio.savemat(mat_file, landmark_dict)
    return landmark_dict


@time_cal
def check_right_eye(points):
    fixed_points = points.copy()
    if points[0][0] < points[1][0]:
        fixed_points[0] = points[4]
        fixed_points[1] = points[3]
        fixed_points[2] = points[2]
        fixed_points[3] = points[1]
        fixed_points[4] = points[0]
        fixed_points[5] = points[7]
        fixed_points[6] = points[6]
        fixed_points[7] = points[5]
    return fixed_points


@time_cal
def check_left_eye(points):
    fixed_points = points.copy()
    if points[0][0] > points[1][0]:
        fixed_points[0] = points[4]
        fixed_points[1] = points[5]
        fixed_points[2] = points[6]
        fixed_points[3] = points[7]
        fixed_points[4] = points[0]
        fixed_points[5] = points[1]
        fixed_points[6] = points[2]
        fixed_points[7] = points[3]
    return fixed_points


@time_cal
def check_face_profile(points):
    # fixed_points = points[16:37]
    # v_x = 2 * points[10][0]
    #
    # left_p = [[v_x - p[0], p[1]] for p in fixed_points[11:][::-1]]
    # right_p = [[v_x - p[0], p[1]] for p in fixed_points[:10][::-1]]
    # merge_p = np.vstack((left_p, fixed_points[10]))
    # merge_p = np.vstack((merge_p, right_p))
    # fixed_points = (fixed_points + merge_p) / 2
    #
    # m1 = get_similarity_matrix(fixed_points, merge_p,True)
    # fixed_points2 = landmark_trans_by_m(points, m1)
    # print(m1)
    return points


@time_cal
def get_points(landmark_dict):
    '''
    :param landmark_dict:
    :return:左眼0-7 左眉8-15 脸16-36 鼻子37-49 嘴50-71 右眉72-79 右眼80-87 88-89左右眼球
    '''

    def _get_eye_center(points):
        eye_center = [(points[0] + points[4])[0] // 2, (points[2] + points[6])[1] // 2]
        return eye_center

    p0 = np.vstack([check_left_eye(landmark_dict['left_eye']), landmark_dict['left_eyebrow']])
    p1 = np.vstack([p0, landmark_dict['face_profile']])
    p2 = np.vstack([p1, landmark_dict['nose']])
    p3 = np.vstack([p2, landmark_dict['mouth']])
    p4 = np.vstack([p3, landmark_dict['right_eyebrow']])
    p5 = np.vstack([p4, check_right_eye(landmark_dict['right_eye'])])
    p6 = np.vstack([p5, [_get_eye_center(landmark_dict['left_eye']), _get_eye_center(landmark_dict['right_eye'])]])
    p6 = check_face_profile(p6)
    return p6, [tuple(p) for p in p6]


@time_cal
def get_similarity_matrix(orange_points, tree_points, fullAffine=False):
    '''
    dst->src 的变换矩阵
    :param dst_points: 目标特征点
    :param src_points: 底图特征点
    :return: matrix
    '''
    m = cv2.estimateRigidTransform(np.array(orange_points), np.array(tree_points), fullAffine)
    if m is None:
        print('异常')
        m = cv2.getAffineTransform(np.float32(orange_points[:3]), np.float32(tree_points[:3]))
    return m


@time_cal
def save_img(img_array, save_name, ifsave):
    if ifsave:
        cv2.imwrite(save_name, img_array)


@time_cal
def landmark_trans_by_m(points, m):
    p1 = np.transpose(points, [1, 0])
    p2 = np.pad(p1, ((0, 1), (0, 0)), 'constant', constant_values=(1, 1))
    p3 = np.matmul(m, p2)
    p4 = np.transpose(p3, [1, 0])
    return p4


@time_cal
def get_measure_triangle():
    triangles = scio.loadmat("triangle_matrix.mat")['triangle']
    return [list(t.astype(np.int32)) for t in triangles]


@time_cal
def get_measure_triangle_skin():
    triangles = scio.loadmat("triangle_matrix_skin_nose.mat")['triangle']
    return [list(t.astype(np.int32)) for t in triangles]


@time_cal
def affine_transform(src, src_tri, dst_tri, size):
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))
    # warp_mat = cv2.estimateRigidTransform(np.array(src_tri), np.array(dst_tri), True)
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]),
                         None,
                         flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


@time_cal
def morph_triangle(src, dst, img, face_mask, t_src, t_dst, t, base_alpha, step=0):
    # t_src, t_dst, t
    # 分别为特征点的三角形坐标

    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    r = cv2.boundingRect(np.float32([t]))
    # 获取三角形的凸包正方形 格式 xmin,ymin,wid,height

    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))
    # 将坐标转换为相对正方形左上角坐标

    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    # 包含剖分三角形的正方形区域
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1., 1., 1.))
    # 填充剖分三角形

    img1_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2_rect = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    size = (r[2], r[3])

    warp_img_src = affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_img_dst = affine_transform(img2_rect, t2_rect, t_rect, size)
    # alpha = 0.5 if step > 49 else alpha
    if step < 16:
        # print('眼睛')
        alpha = min(1.25 * base_alpha, 1.0)
    elif step < 28:
        # print('鼻子')
        alpha = min(1.0 * base_alpha, 1.0)
    elif step < 40:
        # print('眉毛')
        alpha = min(1.13 * base_alpha, 1.0)
    elif step < 50:
        # print('眉毛')
        alpha = min(1.25 * base_alpha, 1.0)
    else:
        alpha = min(1.0 * base_alpha, 1.0)
    img_rect = (1.0 - alpha) * warp_img_src + alpha * warp_img_dst
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + img_rect * mask

    face_mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = face_mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
            1 - mask[:, :, 0]) + 255 * mask[:, :, 0]
    return img, face_mask


@time_cal
def affine_triangle(src, src2, dst, dst2, t_src, t_dst):
    r1 = cv2.boundingRect(np.float32([t_src]))
    r2 = cv2.boundingRect(np.float32([t_dst]))
    t1_rect = []
    t2_rect = []
    t2_rect_int = []

    for i in range(0, 3):
        t1_rect.append((t_src[i][0] - r1[0], t_src[i][1] - r1[1]))
        t2_rect.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))
        t2_rect_int.append((t_dst[i][0] - r2[0], t_dst[i][1] - r2[1]))

    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0))
    img1_rect = src[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    size = (r2[2], r2[3])
    if src2:
        alpha_img1_rect = src2[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        alpha_img2_rect = affine_transform(alpha_img1_rect, t1_rect, t2_rect, size)
        alpha_img2_rect = alpha_img2_rect * mask

    img2_rect = affine_transform(img1_rect, t1_rect, t2_rect, size)
    img2_rect = img2_rect * mask
    # (1620, 280, 3)
    # (800, 0, 820, 1620)
    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect

    if dst2:
        dst2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
                (1.0, 1.0, 1.0) - mask)

        dst2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = dst2[r2[1]:r2[1] + r2[3],
                                                         r2[0]:r2[0] + r2[2]] + alpha_img2_rect


@time_cal
def morph_img(tree_img, tree_points, orange_img, orange_points, alpha):
    def _get_morph_points(_tree_points, _orange_points, alphas):
        '''
        :param src_points:
        :param dst_points:
        :param alphas: eye_alpha, face_alpha, other_alpha  分别为dst 占据的比例
        :return:
        '''
        eye_alpha, face_alpha, other_alpha = alphas
        _morph_points = (1 - other_alpha) * _tree_points + other_alpha * _orange_points
        other_alpha2 = .5
        _mask_points = (1 - other_alpha2) * _tree_points + other_alpha2 * _orange_points

        eye_points = (1 - eye_alpha) * _tree_points + eye_alpha * _orange_points
        face_points = (1 - face_alpha) * _tree_points + face_alpha * _orange_points

        m1 = get_similarity_matrix(_morph_points[0:8] - _morph_points[88], eye_points[0:8] - eye_points[88])
        _morph_points[0:8] = landmark_trans_by_m(_morph_points[0:8] - _morph_points[88], m1) + _morph_points[88]
        m2 = get_similarity_matrix(_morph_points[80:88] - _morph_points[89], eye_points[80:88] - eye_points[89])
        _morph_points[80:88] = landmark_trans_by_m(_morph_points[80:88] - _morph_points[89], m2) + _morph_points[89]

        m3 = get_similarity_matrix(_morph_points[16:37] - _morph_points[26], face_points[16:37] - face_points[26])
        _morph_points[16:37] = landmark_trans_by_m(_morph_points[16:37] - _morph_points[26], m3) + _morph_points[26]

        return _mask_points, _morph_points,

    tree_img = tree_img.astype(np.float32)
    orange_img = orange_img.astype(np.float32)

    res_img = np.zeros(tree_img.shape, dtype=tree_img.dtype)
    _face_mask = np.zeros(orange_img.shape[:2], dtype=np.uint8)

    mask_points, morph_points_ = _get_morph_points(tree_points, orange_points, alpha[:3])
    # morph_points = dst_points
    # src_point 格式[(),()]
    # 根据88点获取149个三角剖分对应的88点的index
    dt = get_measure_triangle()[47:]
    for i in range(0, len(dt)):
        t1 = []
        t2 = []
        t = []
        for j in range(0, 3):
            t1.append(tree_points[dt[i][j]])
            t2.append(orange_points[dt[i][j]])
            t.append(mask_points[dt[i][j]])

        _, face_maskk = morph_triangle(tree_img, orange_img, res_img, _face_mask, t1, t2, t, alpha[3], i)

    return res_img, morph_points_, face_maskk


@time_cal
def tran_src(tree_img, alpha_tree_img, tree_points, orange_points):
    """
    应用三角仿射转换将模板图人脸轮廓仿射成目标图像人脸轮廓
    :param src_img:
    :param src_points:
    :param dst_points:
    :param face_area:
    :return:
    """
    h, w, c = tree_img.shape
    h -= 1
    w -= 1
    mask_area = cv2.boundingRect(np.float32([orange_points]))
    start_x = max(.9 * mask_area[0], 1)
    start_y = max(.9 * mask_area[1], 1)
    end_x = min(start_x + 1.2 * mask_area[2], w - 10)
    end_y = min(start_y + 1.2 * mask_area[3], h - 10)

    sum_x = start_x + end_x
    sum_y = start_y + end_y
    bound_area = np.int32([
        [start_x, start_y], [end_x, start_y], [end_x, end_y], [start_x, end_y],
        [0, 0], [w, 0], [w, h], [0, h],
        [0.5 * sum_x, start_y], [end_x, 0.5 * sum_y], [0.5 * sum_x, end_y], [start_x, 0.5 * sum_y]
    ])

    tree_list = np.vstack([tree_points, bound_area])
    orange_list = np.vstack([orange_points, bound_area])
    res_img = np.zeros(tree_img.shape, dtype=tree_img.dtype)
    alpha_res_img = np.zeros(alpha_tree_img.shape, dtype=alpha_tree_img.dtype) if alpha_tree_img else ''
    dt = get_measure_triangle()
    for i in range(0, len(dt)):
        t_src = []
        t_dst = []

        for j in range(0, 3):
            t_src.append(tree_list[dt[i][j]])
            t_dst.append(orange_list[dt[i][j]])

        affine_triangle(tree_img, alpha_tree_img, res_img, alpha_res_img, t_src, t_dst)

    return res_img, alpha_res_img


@time_cal
def merge_img(orange_img, tree_img, face_mask, orange_points, mat_rate=.88):
    r = cv2.boundingRect(np.float32([orange_points]))

    center = (r[0] + int(r[2] / 2), r[1] + int(int(r[3] / 2)))

    mat = cv2.getRotationMatrix2D(center, 0, mat_rate)
    face_mask = cv2.warpAffine(face_mask, mat, (face_mask.shape[1], face_mask.shape[0]))
    # face_mask = cv2.blur(face_mask, (3, 3))
    # face_mask = cv2.GaussianBlur(face_mask, (27, 27), 1)
    # kernel = np.ones((60, 60), np.uint8)
    # face_mask = cv2.dilate(face_mask, kernel)  # 膨胀
    # face_mask = cv2.erode(face_mask, kernel)  # 腐蚀
    # face_mask = cv2.medianBlur(face_mask, 19)

    return cv2.seamlessClone(np.uint8(orange_img), np.uint8(tree_img), face_mask, center, 1)


@time_cal
def toushi_img(orange_img, orange_points, tree_points, yaw=0):
    if abs(yaw) <= 5:
        rate = 0.1
    else:
        rate = min(abs(yaw), 12) / 12
    _tree = rate * tree_points + (1 - rate) * orange_points
    pts1 = np.float32([orange_points[17], orange_points[18], orange_points[34], orange_points[35]])
    pts2 = np.float32([_tree[17], _tree[18], _tree[34], _tree[35]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    p2 = np.pad(orange_points, ((0, 0), (0, 1)), 'constant', constant_values=(1, 1))
    new_data1 = np.matmul(p2, M.T)
    new_data1 = new_data1 / np.repeat(new_data1[:, 2:3], 3, axis=1)
    new_orange_points = new_data1[:, :2]
    new_orange_img = cv2.warpPerspective(orange_img, M, (2 * orange_img.shape[1], 2 * orange_img.shape[0]))
    return new_orange_img, new_orange_points


@time_cal
def resize_img(img_array, fusion_face_wid):
    img_array = img_array[..., [2, 1, 0, 3]]
    img = Image.fromarray(np.uint8(img_array), "RGBA")
    wid, hei = img.size
    std_face_wid = 257
    fixed_loc = [500, 500]
    # rate = std_face_wid / fusion_face_wid
    # 可优化更合理的对比指标
    rate = max(0.93, std_face_wid / fusion_face_wid)
    img = img.resize([int(rate * wid), int(rate * hei)])
    wid2, hei2 = img.size
    diff_x = abs(int((rate - 1) * fixed_loc[0]))
    diff_y = abs(int((rate - 1) * fixed_loc[1]))
    if wid2 <= wid:
        rr = ((diff_y, wid - wid2 - diff_y), (diff_x, wid - wid2 - diff_x), (0, 0))
        image = np.pad(np.array(img), rr, mode='constant', constant_values=(0, 0))
        img = Image.fromarray(np.uint8(image))
    else:
        img = img.crop([diff_x, diff_y, diff_x + wid, diff_y + hei])

    return img


@time_cal
def get_data_analysis(skin_ori):
    skin_ori_flatten = skin_ori.reshape([-1, 1])
    skin_ori_index = np.flatnonzero(skin_ori_flatten != 0)
    skin_ori_value = skin_ori_flatten[skin_ori_index]

    skin_ori_value_max = np.max(skin_ori_value)
    skin_ori_value_std = np.std(skin_ori_value)
    skin_ori_value_min = np.min(skin_ori_value)
    skin_ori_value_mean = np.mean(skin_ori_value)

    return skin_ori_value_mean, skin_ori_value_std, skin_ori_value_max, skin_ori_value_min


def make_mask(face_mask, t):
    #  t
    # 分别为特征点的三角形坐标
    r = cv2.boundingRect(np.float32([t]))
    # 获取三角形的凸包正方形 格式 xmin,ymin,wid,height

    t_rect = []
    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))

    # 将坐标转换为相对正方形左上角坐标
    mask = np.zeros((r[3], r[2]), dtype=np.float32)
    # 包含剖分三角形的正方形区域
    cv2.fillConvexPoly(mask, np.int32(t_rect), 1)
    # 填充剖分三角形

    face_mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = face_mask[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (
            1 - mask) + 1 * mask
    return face_mask


def get_data_analysis(skin_ori):
    skin_ori_flatten = skin_ori.reshape([-1, 1])
    skin_ori_index = np.flatnonzero(skin_ori_flatten != 0)
    skin_ori_value = skin_ori_flatten[skin_ori_index]

    skin_ori_value_max = np.max(skin_ori_value)
    skin_ori_value_std = np.std(skin_ori_value)
    skin_ori_value_min = np.min(skin_ori_value)
    skin_ori_value_mean = np.mean(skin_ori_value)

    return skin_ori_value_mean, skin_ori_value_std, skin_ori_value_max, skin_ori_value_min


def smooth_light(orange_img, arr_point_tree):
    # 肤色区域
    dt = get_measure_triangle_skin()[47:]
    face_mask2 = np.zeros(orange_img.shape[:2], dtype=np.uint8)
    for i in range(0, len(dt)):
        t = []
        for j in range(0, 3):
            t.append(arr_point_tree[dt[i][j]])

        face_mask = make_mask(face_mask2, t)
    face_mask = np.array(face_mask, np.float32)
    orange_img_hsv = cv2.cvtColor(orange_img, cv2.COLOR_BGR2HSV)

    s = np.array(orange_img_hsv[:, :, 1], np.float32)
    v = np.array(orange_img_hsv[:, :, 2], np.float32)

    s_skin_ori = s * face_mask
    v_skin_ori = v * face_mask

    s_skin_ori_value_mean, s_skin_ori_value_std, s_skin_ori_value_max, s_skin_ori_value_min = get_data_analysis(
        s_skin_ori)

    v_skin_ori_value_mean, v_skin_ori_value_std, v_skin_ori_value_max, v_skin_ori_value_min = get_data_analysis(
        v_skin_ori)

    # 去除不均匀
    # res_img_h = np.clip((h - h_skin_ori_value_mean) / h_skin_ori_value_std * 20 + 50, 16, 230)
    res_img_s = np.clip((s - s_skin_ori_value_mean) / s_skin_ori_value_std * 20 + .95 * s_skin_ori_value_mean, 16, 250)
    res_img_v = np.clip(
        (v - v_skin_ori_value_mean) / v_skin_ori_value_std * .8 * v_skin_ori_value_std + .95 * v_skin_ori_value_mean,
        16, 250)

    # 解决太均匀
    # res_img_s = np.clip(1.1 * res_img_s, 18, 250)
    # res_img_v = np.clip(1.1 * res_img_v, 18, 250)

    # 赋值回原图
    orange_img_hsv[:, :, 1] = res_img_s
    orange_img_hsv[:, :, 2] = res_img_v

    orange_img_hsv2 = cv2.cvtColor(orange_img_hsv, cv2.COLOR_HSV2BGR)

    # 组合成最终图片
    orange_img_hsv = orange_img * (1 - face_mask[:, :, None]) + orange_img_hsv2 * face_mask[:, :, None]
    return np.uint8(orange_img_hsv)


@time_cal
def fusion(orange_path, orange_dict, temp_id, ifsave=True):
    file_name = os.path.basename(orange_path).split('.')[0]
    tree_file = "{}/Templates/{}/ori.jpg".format(Desktop, temp_id)
    landmark_dict_tree = get_landmark_dict(tree_file)

    arr_point_tree, list_point_tree = get_points(landmark_dict_tree)
    tree_left_eye_center = arr_point_tree[88]
    tree_right_eye_center = arr_point_tree[89]
    # tree2 = cv2.imread(tree_file, cv2.IMREAD_UNCHANGED)
    tree = cv2.imread(tree_file, cv2.IMREAD_COLOR)
    tree_center = (tree_right_eye_center + tree_left_eye_center) / 2
    tree_eye_dis = (tree_right_eye_center - tree_left_eye_center)[0]
    # ---------------------------------------------------------#
    # landmark_dict_orange = get_landmark_dict(orange_path)
    # landmark_dict_orange = orange_dict
    arr_point_orange, list_point_orange = get_points(orange_dict)
    orange = cv2.imread(orange_path, cv2.IMREAD_COLOR)
    # from script.mask_face_mask import kk
    # orange = kk(orange_path)
    orange = smooth_light(orange, arr_point_orange)
    save_img(orange, '2-toushied_orange.png', ifsave)
    # orange = cv2.cvtColor(orange, cv2.COLOR_BGR2HSV)
    # orange[:, :, 1] = np.uint8(np.clip(1.1 * np.array(orange[:, :, 1], np.float32), 10, 250))
    # orange[:, :, 2] = np.uint8(np.clip(1.1 * np.array(orange[:, :, 2], np.float32), 10, 250))
    # orange = cv2.cvtColor(orange, cv2.COLOR_HSV2BGR)

    orange, arr_point_orange = toushi_img(orange, arr_point_orange, arr_point_tree, yaw=orange_dict['yaw'])
    save_img(orange, '2-toushied_orange.png', ifsave)
    # arr_point_orange 90*2
    orange_left_eye_center = arr_point_orange[88]
    orange_right_eye_center = arr_point_orange[89]
    orange_center = (orange_right_eye_center + orange_left_eye_center) / 2
    orange_eye_dis = (orange_right_eye_center - orange_left_eye_center)[0]
    # ---------------------------------------------------------#
    # 矫正orange位置与tree对齐

    orange2tree_matrix = get_similarity_matrix(
        orange_points=[orange_left_eye_center, orange_right_eye_center,
                       [orange_center[0], orange_center[1] + orange_eye_dis],
                       [orange_center[0], orange_center[1] - orange_eye_dis]],
        tree_points=[tree_left_eye_center, tree_right_eye_center,
                     [tree_center[0], tree_center[1] + tree_eye_dis],
                     [tree_center[0], tree_center[1] - tree_eye_dis]], fullAffine=False)

    # 矫正后的orange图
    orange_trans = cv2.warpAffine(orange, orange2tree_matrix, (tree.shape[1], tree.shape[0]))
    save_img(orange_trans, '3-orange_trans.png'.format(file_name), ifsave)

    # 矫正后的orange特征点
    arr_point_orange_trans = landmark_trans_by_m(arr_point_orange, orange2tree_matrix)

    # 将orange目标区域扣取1出来，进行比例重组
    orange_mask_trans, morph_points, orange_mask = morph_img(tree, arr_point_tree, orange_trans, arr_point_orange_trans,
                                                             alpha=[.2, .2, .2, .85])  # 眼睛，脸，other

    save_img(orange_mask, '4-orange_mask.png'.format(file_name), ifsave)
    save_img(orange_mask_trans, '4-orange_mask_trans.png'.format(file_name), ifsave)
    # 将Tree进行形变（主要是脸型轮廓）
    tree_trans, alpha_tree_trans = tran_src(tree, '', arr_point_tree, morph_points)
    save_img(tree_trans, '5-tree_trans.png'.format(file_name), ifsave)

    rgb_img = merge_img(orange_mask_trans, np.uint8(tree_trans), orange_mask, morph_points, .88)
    # rgb_img = merge_img(orange_mask_trans, np.uint8(rgb_img), orange_mask, morph_points, .8)
    # save_img(orange_mask, '6-tree_trans.png'.format(file_name), ifsave)

    return rgb_img


if __name__ == '__main__':
    root_dir = os.path.join(Desktop, "Templates", "test_samples")
    test_file = os.path.join(root_dir, '9.png')
    landmark_dict_orange = get_landmark_dict(test_file)

    for i in range(8, 14):
        temp_id = 'temp' + str(i)

        res = fusion(test_file, landmark_dict_orange, temp_id, True)

        save_path = os.path.join(root_dir, '9-{}.jpg'.format(temp_id))
        save_img(res, save_path, True)
