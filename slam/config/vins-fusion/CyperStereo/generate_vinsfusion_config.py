import re
import os
import argparse

def extract_floats(text):
    """从文本中提取所有浮点数（支持科学计数法）"""
    return [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+", text)]

def process_calibration_files(txt_file, vins_yaml, cam0_yaml, cam1_yaml):
    # 检查文件是否存在
    for f in [txt_file, vins_yaml, cam0_yaml, cam1_yaml]:
        if not os.path.exists(f):
            print(f"错误: 找不到文件 {f}")
            return

    with open(txt_file, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    # 清理复制文本时可能带入的干扰符号 (如 )
    #txt_content = re.sub(r'\', '', txt_content)

    # ================= 1 & 2. 提取 T_ic 矩阵 =================
    def get_matrix(target_name):
        match = re.search(rf"{target_name}:.*?\n(.*?)\]\]", txt_content, re.DOTALL)
        if match:
            mat_str = match.group(1) + "]]" 
            return extract_floats(mat_str)
        return []

    t_ic_0 = get_matrix(r"T_ic:\s*\(cam0 to imu0\)")
    t_ic_1 = get_matrix(r"T_ic:\s*\(cam1 to imu0\)")

    # ================= 3 ~ 8. 提取相机内外参 =================
    def get_cam_params(cam_name):
        block_match = re.search(rf"{cam_name}\n-----(.*?)(?=\n\n|\Z)", txt_content, re.DOTALL)
        if not block_match: return {}
        block = block_match.group(1)

        focal_str = re.search(r"Focal length:\s*\[(.*?)\]", block).group(1)
        princ_str = re.search(r"Principal point:\s*\[(.*?)\]", block).group(1)
        dist_str = re.search(r"Distortion coefficients:\s*\[(.*?)\]", block).group(1)

        focals = extract_floats(focal_str)
        princs = extract_floats(princ_str)
        dists = extract_floats(dist_str)

        return {
            'mu': focals[0], 'mv': focals[1],
            'u0': princs[0], 'v0': princs[1],
            'k2': dists[0], 'k3': dists[1], 'k4': dists[2], 'k5': dists[3]
        }

    cam0_params = get_cam_params("cam0")
    cam1_params = get_cam_params("cam1")

    # ================= 执行写入: VINS Fusion Config =================
    with open(vins_yaml, 'r', encoding='utf-8') as f:
        vins_content = f.read()

    def format_mat_for_yaml(mat):
        lines = [", ".join([f"{x:.8f}" for x in mat[i:i+4]]) for i in range(0, 16, 4)]
        return "[" + ",\n           ".join(lines) + "]"

    if len(t_ic_0) == 16:
        vins_content = re.sub(r"(body_T_cam0:.*?data:\s*)\[.*?\]", rf"\g<1>{format_mat_for_yaml(t_ic_0)}", vins_content, flags=re.DOTALL)
    if len(t_ic_1) == 16:
        vins_content = re.sub(r"(body_T_cam1:.*?data:\s*)\[.*?\]", rf"\g<1>{format_mat_for_yaml(t_ic_1)}", vins_content, flags=re.DOTALL)

    with open(vins_yaml, 'w', encoding='utf-8') as f:
        f.write(vins_content)
    print(f"[{vins_yaml}] -> T_ic 矩阵已更新。")

    # ================= 执行写入: Camera Configs =================
    def update_cam_yaml(yaml_path, params):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        for key, val in params.items():
            content = re.sub(rf"(\b{key}:\s*)[-\d.e]+", rf"\g<1>{val}", content)
            
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)

    if cam0_params:
        update_cam_yaml(cam0_yaml, cam0_params)
        print(f"[{cam0_yaml}] -> 内参及畸变系数已更新。")
    if cam1_params:
        update_cam_yaml(cam1_yaml, cam1_params)
        print(f"[{cam1_yaml}] -> 内参及畸变系数已更新。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动将 Kalibr 标定结果覆盖至 VINS-Fusion 配置文件")
    
    # 定义命令行参数，设置默认值为原先的文件名以保持兼容性
    parser.add_argument('--calib_txt', type=str, default='cyperstereo_imu_calibra-results-imucam.txt', help='Kalibr 输出的标定 txt 文件路径')
    parser.add_argument('--vins_yaml', type=str, default='cyperstereo_vins_fusion_config.yaml', help='VINS-Fusion 融合配置文件路径')
    parser.add_argument('--cam0_yaml', type=str, default='cam0_fisheye.yaml', help='cam0 配置文件路径')
    parser.add_argument('--cam1_yaml', type=str, default='cam1_fisheye.yaml', help='cam1 配置文件路径')
    
    args = parser.parse_args()
    
    process_calibration_files(args.calib_txt, args.vins_yaml, args.cam0_yaml, args.cam1_yaml)
