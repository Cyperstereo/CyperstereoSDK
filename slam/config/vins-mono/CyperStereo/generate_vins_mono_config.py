# -*- coding: utf-8 -*-
import os
import re
import argparse
import io

def update_vins_config_inplace(txt_path, yaml_path):
    # 1. 检查输入文件是否存在
    if not os.path.exists(txt_path):
        raise FileNotFoundError(u"错误: 找不到标定结果文件 " + txt_path)
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(u"错误: 找不到原始YAML配置文件 " + yaml_path)

    # ================= 读取标定结果文件 =================
    with io.open(txt_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()

    # 提取外参矩阵 T_ic (cam0 to imu0)
    t_ic_match = re.search(r"T_ic:\s*\(cam0 to imu0\):\s*\[\[(.*?)\]\n\s*\[(.*?)\]\n\s*\[(.*?)\]", txt_content)
    if not t_ic_match:
        raise ValueError(u"未能找到 T_ic: (cam0 to imu0) 矩阵！")

    row0 = [float(x) for x in t_ic_match.group(1).replace('[', '').replace(']', '').split()]
    row1 = [float(x) for x in t_ic_match.group(2).replace('[', '').replace(']', '').split()]
    row2 = [float(x) for x in t_ic_match.group(3).replace('[', '').replace(']', '').split()]

    # 提取左上角 3x3 旋转矩阵
    R_flat = [row0[0], row0[1], row0[2], row1[0], row1[1], row1[2], row2[0], row2[1], row2[2]]
    R_str = ", ".join(["{:.8f}".format(x) for x in R_flat])

    # 提取右上角 3x1 平移向量
    T_flat = [row0[3], row1[3], row2[3]]
    T_str = ", ".join(["{:.8f}".format(x) for x in T_flat])

    # 提取 cam0 内参
    cam0_section = txt_content.split("cam0\n-----")[1].split("cam1\n-----")[0]
    
    focal_match = re.search(r"Focal length:\s*\[(.*?),\s*(.*?)\]", cam0_section)
    mu, mv = focal_match.group(1).strip(), focal_match.group(2).strip()
    
    pp_match = re.search(r"Principal point:\s*\[(.*?),\s*(.*?)\]", cam0_section)
    u0, v0 = pp_match.group(1).strip(), pp_match.group(2).strip()
    
    dist_match = re.search(r"Distortion coefficients:\s*\[(.*?),\s*(.*?),\s*(.*?),\s*(.*?)\]", cam0_section)
    k2, k3, k4, k5 = [dist_match.group(i).strip() for i in range(1, 5)]

    # ================= 读取原始 YAML (读完后会自动关闭句柄) =================
    with io.open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()

    # ================= 在内存中进行正则替换 =================
    yaml_content = re.sub(r"(extrinsicRotation:.*?data:\s*\[).*?(\])", r"\g<1>" + R_str + r"\2", yaml_content, flags=re.DOTALL)
    yaml_content = re.sub(r"(extrinsicTranslation:.*?data:\s*\[).*?(\])", r"\g<1>" + T_str + r"\2", yaml_content, flags=re.DOTALL)
    yaml_content = re.sub(r"(k2:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + k2, yaml_content)
    yaml_content = re.sub(r"(k3:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + k3, yaml_content)
    yaml_content = re.sub(r"(k4:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + k4, yaml_content)
    yaml_content = re.sub(r"(k5:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + k5, yaml_content)
    yaml_content = re.sub(r"(mu:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + mu, yaml_content)
    yaml_content = re.sub(r"(mv:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + mv, yaml_content)
    yaml_content = re.sub(r"(u0:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + u0, yaml_content)
    yaml_content = re.sub(r"(v0:\s*)-?\d+\.?\d*(e-?\d+)?", r"\g<1>" + v0, yaml_content)

    # ================= 原地覆盖写入同一个 YAML 文件 =================
    with io.open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(u"成功: 标定参数已直接覆盖写入原始文件: " + yaml_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="提取 Kalibr 结果并直接原地覆盖 VINS-Mono 配置文件。")
    
    parser.add_argument("--in_txt", type=str, required=True, help="输入的标定结果 txt 文件路径")
    parser.add_argument("--in_yaml", type=str, required=True, help="输入的原始 yaml 配置文件路径 (该文件将被直接修改)")
    
    args = parser.parse_args()

    # 执行原地更新
    update_vins_config_inplace(txt_path=args.in_txt, yaml_path=args.in_yaml)
