from TIN import TIN
import numpy as np
import time
import laspy
import open3d as o3d
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import scipy.linalg

def read(las_file_path, nb_neighbors=30, std_ratio=3.2):
    las = laspy.read(las_file_path)
    points = np.vstack((las.x, las.y, las.z)).transpose()
    PC = o3d.geometry.PointCloud()
    PC.points = o3d.utility.Vector3dVector(points)
    tree = o3d.geometry.KDTreeFlann(PC)

    cl, ind = PC.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    filtered_points = np.asarray(cl.points)
    PC_filtered = o3d.geometry.PointCloud()
    PC_filtered.points = o3d.utility.Vector3dVector(filtered_points)
    return PC_filtered.points, PC_filtered, points

def find_dense_regions(data, window_size, threshold):
    dense_points = []
    for i in range(len(data) - window_size + 1):
        window = data[i:i + window_size]
        if np.std(window) < threshold:
            dense_points.append(window)
    return dense_points

def HashGrid_8_19(GridScale, PC, NI):

    start_time = time.time()
    min_bound, max_bound = PC.get_min_bound(), PC.get_max_bound()
    X_Max, X_Min = max_bound[0], min_bound[0]
    Y_Max, Y_Min = max_bound[1], min_bound[1]

    RowNum = int((X_Max - X_Min) / GridScale) + 1
    ColNum = int((Y_Max - Y_Min) / GridScale) + 1

    Point2dHash = {}
    no_empty_List = set()
    for point in PC.points:
        row = int((point[0] - X_Min) / GridScale)
        col = int((point[1] - Y_Min) / GridScale)
        TempIndex = row * ColNum + col
        if TempIndex not in no_empty_List:
            no_empty_List.add(TempIndex)
        if TempIndex not in Point2dHash:
            Point2dHash[TempIndex] = []
        Point2dHash[TempIndex].append(point)

    elevation_points_dict = {}
    high = np.zeros((RowNum, ColNum))
    for i in range(RowNum):
        for j in range(ColNum):
            TempIndex = i * ColNum + j
            if TempIndex in no_empty_List:
                points_in_grid = np.array(Point2dHash[TempIndex])

                sorted_points = points_in_grid[points_in_grid[:, 2].argsort()]

                if len(sorted_points) > 1:
                    diffs = np.diff(sorted_points[:, 2])
                    valid_indices = np.where(diffs <= 2)[0]
                    selected_idx = valid_indices[0] if len(valid_indices) > 0 else len(sorted_points) - 1
                else:
                    selected_idx = 0

                selected_point = sorted_points[selected_idx]
                high[i][j] = selected_point[2]
                elevation_points_dict[(i, j)] = selected_point
            else:
                high[i][j] = 0

    extended_RowNum = RowNum + NI * 2
    extended_ColNum = ColNum + NI * 2

    extended_high = np.zeros((extended_RowNum, extended_ColNum))
    extended_elevation_points_dict = {}
    sx = [-1, -1, -1, 0, 0, 1, 1, 1]
    sy = [-1, 0, 1, -1, 1, -1, 0, 1]

    for i in range(RowNum):
        for j in range(ColNum):
            extended_high[i + NI, j + NI] = high[i, j]
            if (i, j) in elevation_points_dict:
                extended_elevation_points_dict[(i + NI, j + NI)] = elevation_points_dict[(i, j)]

    extended_high_new = np.zeros((extended_RowNum, extended_ColNum))
    extended = {}

    window_size = int(GridScale / 5) + 1
    for i in range(extended_RowNum):
        for j in range(extended_ColNum):
            if extended_high[i, j] == 0:

                neighbor_coords = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high[ni, nj] != 0:

                            neighbor_coords.append((ni, nj))
                if len(neighbor_coords) >= 1:
                    TEM = [(i - 2) * ColNum + (j - 2) for i, j in neighbor_coords]
                    a = []
                    for TEM_index in TEM:
                        a.extend(Point2dHash[TEM_index])
                    A = np.array(a)

                    x_prime = X_Min + (i - NI + 0.5) * GridScale
                    y_prime = Y_Min + (j - NI + 0.5) * GridScale

                    target_point = np.array([[x_prime, y_prime]])
                    distances = distance.cdist(A[:, :2], target_point, 'euclidean').flatten()
                    average_distance = np.mean(distances)
                    mask = (distances <= average_distance)
                    distances = distances[mask]
                    nig_points = A[:, 2][mask]

                    if len(nig_points) == 0:
                        print()
                    nig_points_sort = sorted(nig_points)
                    if len(nig_points_sort) == 1:
                        dense_regions = nig_points_sort
                        extended_high_new[i, j] = dense_regions[0]
                        extended[(i, j)] = (x_prime, y_prime, dense_regions[0])

                    else:
                        dense_regions = find_dense_regions(nig_points_sort, window_size, 0.6)
                        if len(dense_regions) != 0:
                            mean_values_min = np.mean(np.mean(dense_regions, axis=1))
                        else:
                            mean_values_min = min(nig_points_sort)
                        extended_high_new[i, j] = mean_values_min
                        extended[(i, j)] = (x_prime, y_prime, mean_values_min)

    for i in range(extended_RowNum - 1, -1, -1):
        for j in range(extended_ColNum - 1, -1, -1):
            if extended_high[i, j] == 0:
                neighbor_heights = []
                neighbor_coords = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high[ni, nj] != 0:
                            neighbor_coords.append((ni, nj))
                if len(neighbor_coords) >= 1:

                    TEM = [(i - 2) * ColNum + (j - 2) for i, j in neighbor_coords]
                    a = []
                    for TEM_index in TEM:
                        a.extend(Point2dHash[TEM_index])
                    A = np.array(a)
                    x_prime = X_Min + (i - NI + 0.5) * GridScale
                    y_prime = Y_Min + (j - NI + 0.5) * GridScale
                    target_point = np.array([[x_prime, y_prime]])
                    distances = distance.cdist(A[:, :2], target_point, 'euclidean').flatten()

                    average_distance = np.mean(distances)
                    mask = (distances <= average_distance)
                    distances = distances[mask]
                    nig_points = A[:, 2][mask]

                    if len(nig_points) == 0:
                        print()
                    nig_points_sort = sorted(nig_points)

                    if len(nig_points_sort) == 1:
                        dense_regions = nig_points_sort
                        extended_high_new[i, j] = dense_regions[0]
                        extended[(i, j)] = (x_prime, y_prime, dense_regions[0])

                    else:
                        dense_regions = find_dense_regions(nig_points_sort, window_size, 0.6)
                        if len(dense_regions) != 0:
                            mean_values_min = np.mean(np.mean(dense_regions, axis=1))
                        else:
                            mean_values_min = min(nig_points_sort)

                        extended_high_new[i, j] = mean_values_min
                        extended[(i, j)] = (x_prime, y_prime, mean_values_min)
    extended_high_update = extended_high + extended_high_new

    for i in range(extended_RowNum):
        for j in range(extended_ColNum):
            if extended_high_update[i, j] == 0:

                neighbor_heights1 = []
                neighbor_coords1 = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high_update[ni, nj] != 0:
                            neighbor_heights1.append(extended_high_update[ni, nj])
                            neighbor_coords1.append((ni, nj))

                if len(neighbor_heights1) >= 1:

                    min_height = min(neighbor_heights1)  # 找到最低高程
                    min_index = neighbor_heights1.index(min_height)  # 获取最低高程的索引
                    extended_high_new[i, j] = min_height
                    x_prime = X_Min + (i - NI + 0.5) * GridScale
                    y_prime = Y_Min + (j - NI + 0.5) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    for i in range(extended_RowNum - 1, -1, -1):
        for j in range(extended_ColNum - 1, -1, -1):
            if extended_high_update[i, j] == 0:

                neighbor_heights1 = []
                neighbor_coords1 = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high_update[ni, nj] != 0:

                            neighbor_heights1.append(extended_high_update[ni, nj])
                            neighbor_coords1.append((ni, nj))

                if len(neighbor_heights1) >= 1:
                    min_height = min(neighbor_heights1)
                    min_index = neighbor_heights1.index(min_height)
                    extended_high_new[i, j] = min_height
                    x_prime = X_Min + (i - NI + 0.5) * GridScale
                    y_prime = Y_Min + (j - NI + 0.5) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    extended_elevation_points_dict.update(extended)

    extended_elevation_points_array = {}
    for i, j in extended_elevation_points_dict:
        te = i * extended_ColNum + j
        extended_elevation_points_array[te] = extended_elevation_points_dict[(i, j)]
    return Point2dHash, extended_elevation_points_array, RowNum, ColNum

def HashGrid_8_11(GridScale, PC, NI):

    start_time = time.time()
    min_bound, max_bound = PC.get_min_bound(), PC.get_max_bound()
    X_Max, X_Min = max_bound[0], min_bound[0]
    Y_Max, Y_Min = max_bound[1], min_bound[1]

    RowNum = int((X_Max - X_Min) / GridScale)
    ColNum = int((Y_Max - Y_Min) / GridScale)

    Point2dHash = {}
    no_empty_List = set()
    for point in PC.points:
        row = int((point[0] - X_Min) / GridScale)
        col = int((point[1] - Y_Min) / GridScale)
        TempIndex = row * ColNum + col
        if TempIndex not in no_empty_List:
            no_empty_List.add(TempIndex)
        if TempIndex not in Point2dHash:
            Point2dHash[TempIndex] = []
        Point2dHash[TempIndex].append(point)
    elevation_points_dict = {}
    high = np.zeros((RowNum, ColNum))


    for i in range(RowNum):
        for j in range(ColNum):
            TempIndex = i * ColNum + j
            if TempIndex in no_empty_List:
                temporaryList = Point2dHash[TempIndex]
                pointNum = len(temporaryList)
                highmin = 10000
                min_point_coordinates = None

                for k in range(pointNum):
                    if temporaryList[k][2] < highmin:
                        highmin = temporaryList[k][2]
                        min_point_coordinates = temporaryList[k]

                high[i][j] = highmin
                if min_point_coordinates is not None:
                    elevation_points_dict[(i, j)] = min_point_coordinates
            else:
                high[i][j] = 0


    extended_RowNum = RowNum + NI * 2
    extended_ColNum = ColNum + NI * 2

    extended_high = np.zeros((extended_RowNum, extended_ColNum))
    extended_elevation_points_dict = {}
    sx = [-1, -1, -1, 0, 0, 1, 1, 1]
    sy = [-1, 0, 1, -1, 1, -1, 0, 1]

    for i in range(RowNum):
        for j in range(ColNum):
            extended_high[i + NI, j + NI] = high[i, j]
            if (i, j) in elevation_points_dict:
                extended_elevation_points_dict[(i + NI, j + NI)] = elevation_points_dict[(i, j)]

    extended_high_new = np.zeros((extended_RowNum, extended_ColNum))
    extended = {}
    for i in range(extended_RowNum):
        for j in range(extended_ColNum):
            if extended_high[i, j] == 0:

                neighbor_heights = []
                neighbor_coords = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high[ni, nj] != 0:
                            neighbor_heights.append(extended_high[ni, nj])
                            neighbor_coords.append((ni, nj))

                if len(neighbor_heights) >= 1:

                    min_height = min(neighbor_heights)

                    min_index = neighbor_heights.index(min_height)

                    extended_high_new[i, j] = min_height
                    x_prime = X_Min + (i - NI + 1) * GridScale
                    y_prime = Y_Min + (j - NI + 1) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    for i in range(extended_RowNum - 1, -1, -1):
        for j in range(extended_ColNum - 1, -1, -1):
            if extended_high[i, j] == 0:
                # 搜索周围的 8 个邻近网格
                neighbor_heights = []
                neighbor_coords = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high[ni, nj] != 0:

                            neighbor_heights.append(extended_high[ni, nj])
                            neighbor_coords.append((ni, nj))

                if len(neighbor_heights) >= 1:
                    min_height = min(neighbor_heights)
                    min_index = neighbor_heights.index(min_height)
                    extended_high_new[i, j] = min_height

                    x_prime = X_Min + (i - NI + 1) * GridScale
                    y_prime = Y_Min + (j - NI + 1) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    extended_high_update = extended_high + extended_high_new
    for i in range(extended_RowNum):
        for j in range(extended_ColNum):
            if extended_high_update[i, j] == 0:
                neighbor_heights1 = []
                neighbor_coords1 = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high_update[ni, nj] != 0:

                            neighbor_heights1.append(extended_high_update[ni, nj])
                            neighbor_coords1.append((ni, nj))

                if len(neighbor_heights1) >= 1:
                    min_height = min(neighbor_heights1)
                    min_index = neighbor_heights1.index(min_height)

                    extended_high_new[i, j] = min_height
                    x_prime = X_Min + (i - NI + 1) * GridScale
                    y_prime = Y_Min + (j - NI + 1) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    for i in range(extended_RowNum - 1, -1, -1):
        for j in range(extended_ColNum - 1, -1, -1):
            if extended_high_update[i, j] == 0:
                neighbor_heights1 = []
                neighbor_coords1 = []
                for k in range(8):
                    ni, nj = i + sx[k], j + sy[k]
                    if 0 <= ni < extended_RowNum and 0 <= nj < extended_ColNum:
                        if extended_high_update[ni, nj] != 0:

                            neighbor_heights1.append(extended_high_update[ni, nj])
                            neighbor_coords1.append((ni, nj))

                if len(neighbor_heights1) >= 1:
                    min_height = min(neighbor_heights1)
                    min_index = neighbor_heights1.index(min_height)

                    extended_high_new[i, j] = min_height

                    x_prime = X_Min + (i - NI + 1) * GridScale
                    y_prime = Y_Min + (j - NI + 1) * GridScale
                    extended[(i, j)] = (x_prime, y_prime, min_height)

    extended_elevation_points_dict.update(extended)

    extended_elevation_points_array = {}
    for i, j in extended_elevation_points_dict:
        te = i * extended_ColNum + j
        extended_elevation_points_array[te] = extended_elevation_points_dict[(i, j)]

    return Point2dHash, extended_elevation_points_array, extended_RowNum, extended_ColNum

def scale_factor(W_min, W_max):
    s = W_max / W_min
    for lay in range(1, 10):
        if 1 <= s ** (1 / lay) <= 1.9:
            break
    d = s ** (1 / lay)

    results = []
    for lay in range(1, lay + 2):
        result = round(W_min * d ** (lay - 1), 1)
        results.append(result)
    results.reverse()
    return results

def Elevation(S_min, S_max, scales):
    elevation_params = []
    for i in range(1, len(scales) + 1):
        elevation_params.append((i - 1)*(S_max-S_min)/(len(scales) - 1)+S_min)
    elevation_params.reverse()
    return elevation_params

def reach_dex(in_points, in_points_dicts):
    points_dicts = {}
    value_to_key = {tuple(value): key for key, value in in_points_dicts.items()}

    for b_item in in_points:
        b_item_tuple = tuple(b_item)
        if b_item_tuple in value_to_key:
            key = value_to_key[b_item_tuple]
            points_dicts[key] = in_points_dicts[key]
    return points_dicts

def Fault_compensation(no_ground_dicts, ground_dicts, ground_points, ColNum, dis):
    ground_keys_set = set(ground_dicts.keys())
    keys_to_delete = []
    for key, value in no_ground_dicts.items():
        elevation_data = [
            ground_dicts[index] for index in [
                key - ColNum - 1, key - ColNum, key - ColNum + 1,
                key - 1, key + 1,
                key + ColNum - 1, key + ColNum, key + ColNum + 1,
            ] if index in ground_keys_set
        ]

        if elevation_data:
            ZD_array = np.array(elevation_data)
            FD_array = np.array(value)
            distances = distance.cdist(ZD_array[:, :2], [FD_array[:2]], 'euclidean').flatten()
            high = FD_array[2] - ZD_array[:, 2]
            slope1 = (high / distances)
            slope_degrees = np.degrees(np.arctan(slope1))

            mask = (distances <= dis * 2.8284) & (high <= 1.4)
            distances = distances[mask]
            high = high[mask]
            slope_degrees = slope_degrees[mask]
            zheng_count = np.sum(high > 0)
            fu_count = np.sum(high <= 0)
            di_count = np.sum(slope_degrees <= 10)
            gao_count = np.sum(slope_degrees > 10)

            if fu_count > 0 and zheng_count <= 2 or gao_count == 0 and di_count >= 3 or (gao_count > 0 and di_count >= 2):
                ground_points.append(no_ground_dicts[key])

                keys_to_delete.append(key)
    for key in keys_to_delete:
        ground_dicts[key] = no_ground_dicts[key]
        del no_ground_dicts[key]

    return

def Valley_compensation(no_ground_dicts, ground_dicts, ground_points, ColNum):
    num = 0
    def tps_kernel(r, epsilon=1e-6):
        r = np.maximum(r, epsilon)
        return r ** 2 * np.log(r)

    def tps_matrix(X, Y):
        dist = cdist(X, Y, metric='euclidean')
        K = tps_kernel(dist)
        return K

    def fit_tps_zhengze(X, Z, lambda_reg):
        n = len(X)
        P = np.hstack((np.ones((n, 1)), X))
        K = tps_matrix(X, X)

        K_reg = K.copy()
        if lambda_reg > 0:
            K_reg += lambda_reg * np.eye(n)

        L = np.vstack((
            np.hstack((K_reg, P)),
            np.hstack((P.T, np.zeros((3, 3))))
        ))

        b = np.concatenate((Z, np.zeros(3)))


        try:
            coeffs = scipy.linalg.solve(L, b, assume_a='sym')
        except scipy.linalg.LinAlgError:
            L_reg = L + 1e-8 * np.eye(L.shape[0])
            coeffs = scipy.linalg.solve(L_reg, b, assume_a='sym')
        return coeffs, K

    def predict_tps(X, X_train, coeffs, K):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != X_train.shape[1]:
            raise ValueError(f"点维度不匹配: 预测点维度{X.shape[1]} vs 训练点维度{X_train.shape[1]}")

        P_new = np.hstack((np.ones((X.shape[0], 1)), X))
        dist_new = cdist(X, X_train, metric='euclidean')
        K_new = tps_kernel(dist_new)
        n_train = X_train.shape[0]
        lambda_coeffs = coeffs[:n_train]
        mu_coeffs = coeffs[n_train:]

        prediction = K_new.dot(lambda_coeffs) + P_new.dot(mu_coeffs)
        return prediction


    for key, value in no_ground_dicts.items():
        nearby_indices = [
            key - 2 * ColNum - 2, key - 2 * ColNum - 1, key - 2 * ColNum, key - 2 * ColNum + 1, key - 2 * ColNum + 2,
            key - ColNum - 2, key - ColNum - 1, key - ColNum, key - ColNum + 1, key - ColNum + 2,
            key - 2, key - 1, key + 1, key + 2,
            key + ColNum - 2, key + ColNum - 1, key + ColNum, key + ColNum + 1, key + ColNum + 2,
            key + 2 * ColNum - 2, key + 2 * ColNum - 1, key + 2 * ColNum, key + 2 * ColNum + 1, key + 2 * ColNum + 2,
        ]

        around_data = [ground_dicts[index] for index in nearby_indices if index in ground_dicts]
        point = no_ground_dicts[key]
        current_point = no_ground_dicts[key]

        filtered_around_data = []
        for point in around_data:
            dx = abs(point[0] - current_point[0])
            dy = abs(point[1] - current_point[1])
            if dx > 50 or dy > 50:
                continue
            filtered_around_data.append(point)

        if len(filtered_around_data) > 3:
            around_array = np.array(filtered_around_data)
            X_train = around_array[:, :2]
            Z_train = around_array[:, 2]
            if not hasattr(Valley_compensation, 'last_norm_params'):
                min_val = np.min(X_train, axis=0)
                max_val = np.max(X_train, axis=0)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1
                Valley_compensation.norm_params = (min_val, range_val)


            min_val, range_val = Valley_compensation.norm_params
            X_train_norm = 2 * (X_train - min_val) / range_val - 1

            coeffs, K = fit_tps_zhengze(X_train_norm, Z_train, 0.2)

            X_test = np.array([point[:2]])

            X_test_norm = 2 * (X_test - min_val) / range_val - 1

            Z_predictions = predict_tps(X_test_norm, X_train_norm, coeffs, K)

            if isinstance(Z_predictions, np.ndarray) and Z_predictions.size > 0:
                Z_pred = Z_predictions[0]
            else:
                continue

            distance = abs(Z_pred - point[2])

            if distance < 0.300:
                num += 1
                ground_points.append(point)

    return ground_points


def Valley_compensation_polynomial(no_ground_dicts, ground_dicts, ground_points, ColNum, degree=2):
    num = 0
    def fit_polynomial_surface(X, Z, degree=2, lambda_reg=1e-6):

        n = len(X)

        A = build_polynomial_features(X, degree)

        if lambda_reg > 0:
            reg_matrix = lambda_reg * np.eye(A.shape[1])
            reg_matrix[0, 0] = 0
        else:
            reg_matrix = np.zeros((A.shape[1], A.shape[1]))
        try:
            coeffs = scipy.linalg.solve(A.T @ A + reg_matrix, A.T @ Z, assume_a='sym')
        except scipy.linalg.LinAlgError:

            coeffs = np.linalg.pinv(A.T @ A + reg_matrix) @ A.T @ Z

        return coeffs

    def build_polynomial_features(X, degree):

        n = len(X)
        features = []

        # 添加常数项
        features.append(np.ones(n))

        # 添加各次项
        for d in range(1, degree + 1):
            for i in range(d + 1):
                j = d - i
                if i == 0:
                    feature = X[:, 1] ** j  # y^j
                elif j == 0:
                    feature = X[:, 0] ** i  # x^i
                else:
                    feature = (X[:, 0] ** i) * (X[:, 1] ** j)  # x^i * y^j
                features.append(feature)

        return np.column_stack(features)

    def predict_polynomial_surface(X, coeffs, degree):
        A = build_polynomial_features(X, degree)
        return A @ coeffs

    for key, value in no_ground_dicts.items():
        nearby_indices = [
            key - 2 * ColNum - 2, key - 2 * ColNum - 1, key - 2 * ColNum,
            key - 2 * ColNum + 1, key - 2 * ColNum + 2,
            key - ColNum - 2, key - ColNum - 1, key - ColNum,
            key - ColNum + 1, key - ColNum + 2,
            key - 2, key - 1, key + 1, key + 2,
            key + ColNum - 2, key + ColNum - 1, key + ColNum,
            key + ColNum + 1, key + ColNum + 2,
            key + 2 * ColNum - 2, key + 2 * ColNum - 1, key + 2 * ColNum,
            key + 2 * ColNum + 1, key + 2 * ColNum + 2,
        ]

        around_data = [ground_dicts[index] for index in nearby_indices if index in ground_dicts]
        current_point = no_ground_dicts[key]

        filtered_around_data = []
        for point in around_data:
            dx = abs(point[0] - current_point[0])
            dy = abs(point[1] - current_point[1])
            if dx <= 50 and dy <= 50:
                filtered_around_data.append(point)

        if len(filtered_around_data) >= 6:
            around_array = np.array(filtered_around_data)
            X_train = around_array[:, :2]
            Z_train = around_array[:, 2]

            if not hasattr(Valley_compensation_polynomial, 'norm_params'):
                min_val = np.min(X_train, axis=0)
                max_val = np.max(X_train, axis=0)
                range_val = max_val - min_val
                range_val[range_val == 0] = 1
                Valley_compensation_polynomial.norm_params = (min_val, range_val)

            min_val, range_val = Valley_compensation_polynomial.norm_params
            X_train_norm = 2 * (X_train - min_val) / range_val - 1
            try:
                coeffs = fit_polynomial_surface(X_train_norm, Z_train, degree=degree, lambda_reg=1e-4)

                X_test = np.array([current_point[:2]])
                X_test_norm = 2 * (X_test - min_val) / range_val - 1

                Z_prediction = predict_polynomial_surface(X_test_norm, coeffs, degree=degree)
                if isinstance(Z_prediction, np.ndarray) and Z_prediction.size > 0:
                    Z_pred = Z_prediction[0]

                    distance = abs(Z_pred - current_point[2])
                    if distance < 0.300:
                        num += 1
                        ground_points.append(current_point)
            except Exception as e:
                continue
    print(f"多项式曲面补偿: 新增 {num} 个地面点")
    return ground_points


def export_ground_points(filtered_points, output_las_file):

    points_array = np.array(filtered_points)
    las_points = laspy.create(point_format=3, file_version="1.2")

    las_points.x = points_array[:, 0]
    las_points.y = points_array[:, 1]
    las_points.z = points_array[:, 2]

    las_points.write(output_las_file)

def function(W_max, S_min, S_max, Angle_min, las_file_path, output_las_file):

    points, PC, points00 = read(las_file_path)

    Dis_min = S_min - 0.2
    scales = scale_factor(2, W_max)

    for i, scale in enumerate(scales):
        Point2dHash = f"Point2dHash{i}"
        elevation_points_dict = f"elevation_points_dict{i}"
        RowNum = f"RowNum{i}"
        ColNum = f"ColNum{i}"

        if i >= len(scales) - 2:
            result = HashGrid_8_11(scale, PC, 2)
        else:
            result = HashGrid_8_19(scale, PC, 2)

        globals()[Point2dHash], globals()[elevation_points_dict], globals()[RowNum], globals()[ColNum] = result

    ground_points0 = [value for value in elevation_points_dict0.values()]
    elevation_params = Elevation(S_min, S_max, scales)

    # 根据 scales 的长度动态执行循环
    for i in range(1, len(scales)):
        # print(f'第{i}层')

        if i == 1:
            ground_points = ground_points0
        else:
            ground_points = globals()[f"GT_{i - 1}"]

        GT_ = f"GT_{i}"  # GT_1, GT_2, GT_3, GT_4, GT_5
        no_GT_ = f"no_GT_{i}"  # no_GT_1, no_GT_2, no_GT_3, no_GT_4, no_GT_5

        elevation_points_dict = f"elevation_points_dict{i}"  # elevation_points_dict1, elevation_points_dict2
        ColNum = f"ColNum{i}"
        ground_points_dict = f"ground_points_dict{i}"
        no_ground_points_dict = f"no_ground_points_dict{i}"

        tin_net = TIN(np.array(ground_points))
        globals()[GT_], globals()[no_GT_] = tin_net.find_elevation(globals()[elevation_points_dict],
                                                                   elevation_params[i - 1], Dis_min=Dis_min,
                                                                   Angle_min=Angle_min, display=False)

        globals()[ground_points_dict] = reach_dex(globals()[GT_], globals()[elevation_points_dict])
        globals()[no_ground_points_dict] = reach_dex(globals()[no_GT_], globals()[elevation_points_dict])

        Fault_compensation(globals()[no_ground_points_dict], globals()[ground_points_dict], globals()[GT_],
                        globals()[ColNum], scales[i - 1])


        Valley_compensation_polynomial(globals()[no_ground_points_dict], globals()[ground_points_dict], globals()[GT_], globals()[ColNum])
        Valley_compensation_polynomial(globals()[no_ground_points_dict], globals()[ground_points_dict], globals()[GT_], globals()[ColNum])
        Valley_compensation_polynomial(globals()[no_ground_points_dict], globals()[ground_points_dict], globals()[GT_], globals()[ColNum])
        Valley_compensation_polynomial(globals()[no_ground_points_dict], globals()[ground_points_dict], globals()[GT_], globals()[ColNum])

        if i == len(scales) - 1:
            last_GT = globals()[f"GT_{i}"]

    tin_net = TIN(np.array(last_GT))
    last_points, no_last_points = tin_net.find_elevation(points00, elevation_params[-1], Dis_min=Dis_min, Angle_min=Angle_min,
                                                         display=False)


    export_ground_points(last_points, output_las_file)

    return last_points, output_las_file


