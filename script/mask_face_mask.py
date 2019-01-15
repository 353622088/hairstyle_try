# coding:utf-8 
'''
created on 2019/1/12

@author:Dxq
'''
from script.main import *


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


def kk(file):
    orange_img = cv2.imread(file)
    landmark_dict_tree = get_landmark_dict(file)
    arr_point_tree, list_point_tree = get_points(landmark_dict_tree)
    dt = get_measure_triangle_skin()[47:]
    face_mask2 = np.zeros(orange_img.shape[:2], dtype=np.uint8)
    for i in range(0, len(dt)):
        t = []
        for j in range(0, 3):
            t.append(arr_point_tree[dt[i][j]])

        face_mask = make_mask(face_mask2, t)
    face_mask = np.array(face_mask, np.float32)

    orange_img_hsv = cv2.cvtColor(orange_img, cv2.COLOR_BGR2HSV)
    h = np.array(orange_img_hsv[:, :, 0], np.float32)
    s = np.array(orange_img_hsv[:, :, 1], np.float32)
    v = np.array(orange_img_hsv[:, :, 2], np.float32)

    h_skin_ori = h * face_mask
    s_skin_ori = s * face_mask
    v_skin_ori = v * face_mask

    # cv2.imwrite("1h_skin_ori.png", h_skin_ori)
    # cv2.imwrite("1s_skin_ori.png", s_skin_ori)
    # cv2.imwrite("1v_skin_ori.png", v_skin_ori)

    # h_skin_ori_value_mean, h_skin_ori_value_std, h_skin_ori_value_max, h_skin_ori_value_min = get_data_analysis(
    #     h_skin_ori)

    s_skin_ori_value_mean, s_skin_ori_value_std, s_skin_ori_value_max, s_skin_ori_value_min = get_data_analysis(
        s_skin_ori)

    v_skin_ori_value_mean, v_skin_ori_value_std, v_skin_ori_value_max, v_skin_ori_value_min = get_data_analysis(
        v_skin_ori)

    # print(h_skin_ori_value_std, h_skin_ori_value_max, h_skin_ori_value_min, h_skin_ori_value_mean)
    # print(s_skin_ori_value_std, s_skin_ori_value_max, s_skin_ori_value_min, s_skin_ori_value_mean)
    # print(v_skin_ori_value_std, v_skin_ori_value_max, v_skin_ori_value_min, v_skin_ori_value_mean)

    # 去除不均匀
    # res_img_h = np.clip((h - h_skin_ori_value_mean) / h_skin_ori_value_std * 20 + 50, 16, 230)
    res_img_s = np.clip((s - s_skin_ori_value_mean) / s_skin_ori_value_std * 20 + .95 * s_skin_ori_value_mean, 16, 250)
    res_img_v = np.clip(
        (v - v_skin_ori_value_mean) / v_skin_ori_value_std * .8 * v_skin_ori_value_std + .95 * v_skin_ori_value_mean,
        16, 250)
    print(v_skin_ori_value_mean, s_skin_ori_value_mean)
    print(s_skin_ori_value_std, v_skin_ori_value_std)
    # 解决太均匀
    # res_img_s = np.clip(1.1 * res_img_s, 18, 250)
    # res_img_v = np.clip(1.1 * res_img_v, 18, 250)

    # cv2.imwrite("1h_skin_ori_step1.png", res_img_h * face_mask)
    # cv2.imwrite("1s_skin_ori_step1.png", res_img_s * face_mask)
    # cv2.imwrite("1v_skin_ori_step1.png", res_img_v * face_mask)

    # diff_v = np.array(res_img_v, np.float32) - 225
    # diff_v_mask = abs(diff_v) < 30 * 3
    # special_v_ori_mask = (1 - diff_v_mask) * face_mask * v
    # v_special_mean, v_special_std, v_special_max, v_special_min = get_data_analysis(special_v_ori_mask)
    # special_v_ori_mask = np.clip((special_v_ori_mask - v_special_mean) / v_special_std * 26 + 225, 16, 230)
    # res_img_v = res_img_v * diff_v_mask + special_v_ori_mask * (1 - diff_v_mask)
    #
    # cv2.imwrite("1special_ori_v_mask.png", special_v_ori_mask)
    # cv2.imwrite("1special_ori_v_mask2.png", special_v_ori_mask)
    # cv2.imwrite("1v_skin_ori_step2.png", res_img_v * face_mask)
    # diff_s = np.array(orange_img_hsv[:, :, 1], np.float32) - 50
    # diff_s_mask = abs(diff_s) < 20 * 3

    orange_img_hsv[:, :, 1] = res_img_s
    orange_img_hsv[:, :, 2] = res_img_v

    orange_img_hsv2 = cv2.cvtColor(orange_img_hsv, cv2.COLOR_HSV2BGR)
    orange_img_hsv = orange_img * (1 - face_mask[:, :, None]) + orange_img_hsv2 * face_mask[:, :, None]
    return np.uint8(orange_img_hsv)

# res = kk("1.png")
# cv2.imwrite("orange_img_hsv1.png", res)
