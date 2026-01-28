import time
import numpy as np
import math
from joblib import Parallel, delayed
import os
from MATPTD import function
import laspy


def process_block(W_max, S_min, S_max, Angle_min, point_cloud_block, index, output_dir):

    print(f"Processing block {index}")
    temp_input_path = f"{output_dir}/temp_input_{index}.las"
    temp_output_path = f"{output_dir}/temp_output_{index}.las"

    save_point_cloud_to_las(point_cloud_block, temp_input_path)
    try:
        last_points, output_las_file = function(
            W_max=W_max,
            S_min=S_min,
            S_max=S_max,
            Angle_min=Angle_min,
            las_file_path=temp_input_path,
            output_las_file=temp_output_path
        )

        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        return last_points, point_cloud_block, index

    except Exception as e:
        print(f"Error processing block {index}: {e}")

        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)
        return None, point_cloud_block, index

def split_point_cloud(point_cloud, bounding_box, block_size, buffer_size):
    xtopleft, ytopleft, xbottomright, ybottomright = bounding_box
    width = xbottomright - xtopleft
    height = ytopleft - ybottomright
    if width <= block_size or height <= block_size:
        print("Point cloud is too small, returning the whole point cloud")
        return [(point_cloud, bounding_box)]
    num_cols = math.ceil(width / block_size)
    num_rows = math.ceil(height / block_size)
    print(f"num_cols={num_cols}, num_rows={num_rows}")
    split_boxes = []
    subclouds = []

    for row in range(num_rows):
        for col in range(num_cols):
            x_min = xtopleft + col * block_size
            x_max = x_min + block_size + buffer_size
            y_max = ytopleft - row * block_size
            y_min = y_max - block_size - buffer_size

            if x_min < xtopleft:
                x_min = xtopleft
            if x_max > xbottomright:
                x_max = xbottomright
            if y_max > ytopleft:
                y_max = ytopleft
            if y_min < ybottomright:
                y_min = ybottomright

            x_mask = (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] < x_max)
            y_mask = (point_cloud[:, 1] > y_min) & (point_cloud[:, 1] <= y_max)

            if col == num_cols - 1:
                x_max = xbottomright
                x_mask = (point_cloud[:, 0] >= x_min) & (point_cloud[:, 0] <= x_max)
            if row == num_rows - 1:
                y_min = ybottomright
                y_mask = (point_cloud[:, 1] >= y_min) & (point_cloud[:, 1] <= y_max)

            mask = x_mask & y_mask
            subcloud = point_cloud[mask]

            if len(subcloud) > 0:
                sub_box = (x_min, y_max, x_max, y_min)
                split_boxes.append(sub_box)
                subclouds.append(subcloud)

    return list(zip(subclouds, split_boxes))

def save_point_cloud_to_las(point_cloud, file_path):

    try:
        header = laspy.LasHeader(point_format=3)
        las = laspy.LasData(header)

        las.x = point_cloud[:, 0]
        las.y = point_cloud[:, 1]
        las.z = point_cloud[:, 2]

        las.write(file_path)
        print(f"点云已保存: {file_path}")

    except Exception as e:
        print(f"保存LAS文件失败: {e}")
        raise

def load_point_cloud_from_las(file_path):
    try:
        print(f"正在加载点云文件: {file_path}")

        las = laspy.read(file_path)
        x = las.x
        y = las.y
        z = las.z

        points = np.vstack((x, y, z)).T
        print(f"点云加载完成: {len(points)} 个点")
        print(f"坐标范围: X({np.min(x):.2f}-{np.max(x):.2f}), "
              f"Y({np.min(y):.2f}-{np.max(y):.2f}), "
              f"Z({np.min(z):.2f}-{np.max(z):.2f})")
        return points
    except Exception as e:
        print(f"加载LAS文件失败: {e}")
        raise


def main():

    W_max = 40
    S_min = 0.5
    S_max = 0.8
    Angle_min = 20

    input_las_file = r"input_pointcloud.las"
    output_dir = r"results/"
    final_output_las = r"ground_points.las"

    block_size = 400
    buffer_size = 0.03*block_size
    os.makedirs(output_dir, exist_ok=True)

    print("Loading point cloud...")
    points = load_point_cloud_from_las(input_las_file)

    x_min, y_min, z_min = np.min(points, axis=0)
    x_max, y_max, z_max = np.max(points, axis=0)
    bounding_box = (x_min, y_max, x_max, y_min)

    print(f"Point cloud loaded: {len(points)} points")
    print(f"Bounding box: {bounding_box}")
    print(f"Block size: {block_size}, Buffer size: {buffer_size}")

    print("Splitting point cloud...")
    split_blocks = split_point_cloud(
        points,
        bounding_box,
        block_size=block_size,
        buffer_size=buffer_size
    )
    print(f"Split into {len(split_blocks)} blocks")

    print("Processing blocks in parallel...")

    results = Parallel(n_jobs=os.cpu_count())(
        delayed(process_block)(
            W_max, S_min, S_max, Angle_min,
            block_pc, idx, output_dir
        )
        for idx, (block_pc, block_bbox) in enumerate(split_blocks)
    )

    print("Merging results...")
    ground_points_list = []

    for result in results:
        last_points, original_pc, idx = result
        if last_points is not None and len(last_points) > 0:
            ground_points_list.append(last_points)

    if ground_points_list:
        ground_points = np.vstack(ground_points_list)
        print(f"Merged ground points: {len(ground_points)} points")

        save_point_cloud_to_las(ground_points, final_output_las)
        print(f"Final ground points saved to: {final_output_las}")

        return ground_points
    else:
        print("No ground points found!")
        return None

if __name__ == '__main__':
    start_time = time.time()

    try:
        ground_points = main()
        if ground_points is not None:
            print(f"Processing completed successfully. Found {len(ground_points)} ground points.")
    except Exception as e:
        print(f"Error in main execution: {e}")

    end_time = time.time()
    delt_time = end_time - start_time
    print(f"Total time: {delt_time:.3f} seconds")