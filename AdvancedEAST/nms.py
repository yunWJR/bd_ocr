# coding=utf-8
import math

import numpy as np

import cfg


def should_merge(region, i, j):
    neighbor = {(i, j - 1)}
    return not region.isdisjoint(neighbor)


def region_neighbor(region_set):
    region_pixels = np.array(list(region_set))
    j_min = np.amin(region_pixels, axis=0)[1] - 1
    j_max = np.amax(region_pixels, axis=0)[1] + 1
    i_m = np.amin(region_pixels, axis=0)[0] + 1
    region_pixels[:, 0] += 1
    neighbor = {(region_pixels[n, 0], region_pixels[n, 1]) for n in
                range(len(region_pixels))}
    neighbor.add((i_m, j_min))
    neighbor.add((i_m, j_max))
    return neighbor


def region_group(region_list):
    S = [i for i in range(len(region_list))]
    D = []
    while len(S) > 0:
        m = S.pop(0)
        if len(S) == 0:
            # S has only one element, put it to D
            D.append([m])
        else:
            D.append(rec_region_merge(region_list, m, S))
    return D


def rec_region_merge(region_list, m, S):
    rows = [m]
    tmp = []
    for n in S:
        if not region_neighbor(region_list[m]).isdisjoint(region_list[n]) or \
                not region_neighbor(region_list[n]).isdisjoint(region_list[m]):
            # 第m与n相交
            tmp.append(n)
    for d in tmp:
        S.remove(d)
    for e in tmp:
        rows.extend(rec_region_merge(region_list, e, S))
    return rows


def nms(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold, fix=False):
    if fix:
        return nmsFix(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold)
    else:
        return nmsOrg(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold)


# 最小二乘拟合直线
def Least_squares(x, y):
    x_ = x.mean()
    y_ = y.mean()
    m = np.zeros(1)
    n = np.zeros(1)
    k = np.zeros(1)
    p = np.zeros(1)
    for i in np.arange(len(x)):
        k = (x[i] - x_) * (y[i] - y_)
        m += k
        p = np.square(x[i] - x_)
        n = n + p
    a = m / n
    b = y_ - a * x_
    return a, b


def nmsFix(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        if len(group) <= 1:
            continue

        cord_list = []
        for row in group:
            for ij in region_list[row]:
                cord_list.append((ij[0], ij[1]))
        cord_list = np.array(cord_list)

        min_i, min_j = np.amin(cord_list, axis=0)
        max_i, max_j = np.amax(cord_list, axis=0)
        quad_list[g_th, 0] = np.array([(min_j - 1) * cfg.pixel_size, (min_i - 1) * cfg.pixel_size])
        quad_list[g_th, 1] = np.array([(min_j - 1) * cfg.pixel_size, (max_i + 1) * cfg.pixel_size])
        quad_list[g_th, 2] = np.array([(max_j + 1) * cfg.pixel_size, (max_i + 1) * cfg.pixel_size])
        quad_list[g_th, 3] = np.array([(max_j + 1) * cfg.pixel_size, (min_i - 1) * cfg.pixel_size])

        a, b = Least_squares(cord_list[:, 0], cord_list[:, 1])

        # 水平
        if a == 0:
            continue

        ymax = ymin = 0

        for cl in cord_list:
            y = a * cl[0] + b
            yd = cl[1] - y
            if yd > ymax:
                ymax = yd

            if yd < ymin:
                ymin = yd

        yv = abs(ymax - ymin)

        alfa = math.atan(abs(a))

        yh = math.sin(alfa) * yv
        yh = abs(yh)

        ydelta = yh / 2 * math.sin(alfa) * cfg.pixel_size
        ydelta = abs(ydelta)

        xdelta = yh / 2 * math.cos(alfa) * cfg.pixel_size
        xdelta = abs(xdelta)

        r_quad = np.zeros((4, 2))
        if a > 0:
            r_quad[0] = np.sum(quad_list[g_th, 0] + np.array([xdelta, -ydelta]), axis=0) / 2
            r_quad[1] = np.sum(quad_list[g_th, 0] + np.array([-xdelta, ydelta]), axis=0) / 2
            r_quad[2] = np.sum(quad_list[g_th, 2] + np.array([-xdelta, ydelta]), axis=0) / 2
            r_quad[3] = np.sum(quad_list[g_th, 2] + np.array([xdelta, -ydelta]), axis=0) / 2
        else:
            r_quad[0] = np.sum(quad_list[g_th, 3] + np.array([-xdelta, -ydelta]), axis=0) / 2
            r_quad[3] = np.sum(quad_list[g_th, 3] + np.array([xdelta, ydelta]), axis=0) / 2
            r_quad[2] = np.sum(quad_list[g_th, 1] + np.array([xdelta, ydelta]), axis=0) / 2
            r_quad[1] = np.sum(quad_list[g_th, 1] + np.array([-xdelta, -ydelta]), axis=0) / 2

        quad_list[g_th] = r_quad

    return score_list, quad_list


# https://blog.csdn.net/linchuhai/article/details/84677249 todo
def nmsFixSimple(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    # TODO(linchuhai):这里确定每个文本框的坐标还需要进一步修改
    for group, g_th in zip(D, range(len(D))):
        cord_list = []
        for row in group:
            for ij in region_list[row]:
                cord_list.append((ij[0], ij[1]))
        cord_list = np.array(cord_list)
        min_i, min_j = np.amin(cord_list, axis=0)
        max_i, max_j = np.amax(cord_list, axis=0)
        quad_list[g_th, 0] = np.array([(min_j - 1) * cfg.pixel_size, (min_i - 1) * cfg.pixel_size])
        quad_list[g_th, 1] = np.array([(min_j - 1) * cfg.pixel_size, (max_i + 1) * cfg.pixel_size])
        quad_list[g_th, 2] = np.array([(max_j + 1) * cfg.pixel_size, (max_i + 1) * cfg.pixel_size])
        quad_list[g_th, 3] = np.array([(max_j + 1) * cfg.pixel_size, (min_i - 1) * cfg.pixel_size])

    return score_list, quad_list


def nmsOrg(predict, activation_pixels, threshold=cfg.side_vertex_pixel_threshold):
    region_list = []
    for i, j in zip(activation_pixels[0], activation_pixels[1]):
        merge = False
        for k in range(len(region_list)):
            if should_merge(region_list[k], i, j):
                region_list[k].add((i, j))
                merge = True
                # Fixme 重叠文本区域处理，存在和多个区域邻接的pixels，先都merge试试
                # break
        if not merge:
            region_list.append({(i, j)})
    D = region_group(region_list)
    quad_list = np.zeros((len(D), 4, 2))
    score_list = np.zeros((len(D), 4))
    for group, g_th in zip(D, range(len(D))):
        total_score = np.zeros((4, 2))
        for row in group:
            for ij in region_list[row]:
                score = predict[ij[0], ij[1], 1]
                if score >= threshold:
                    ith_score = predict[ij[0], ij[1], 2:3]
                    if not (cfg.trunc_threshold <= ith_score < 1 - cfg.trunc_threshold):
                        ith = int(np.around(ith_score))
                        total_score[ith * 2:(ith + 1) * 2] += score
                        px = (ij[1] + 0.5) * cfg.pixel_size
                        py = (ij[0] + 0.5) * cfg.pixel_size
                        p_v = [px, py] + np.reshape(predict[ij[0], ij[1], 3:7], (2, 2))
                        quad_list[g_th, ith * 2:(ith + 1) * 2] += score * p_v
        score_list[g_th] = total_score[:, 0]
        quad_list[g_th] /= (total_score + cfg.epsilon)
    return score_list, quad_list
