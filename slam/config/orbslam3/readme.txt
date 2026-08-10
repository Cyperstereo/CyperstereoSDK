#python3 generate_cyperstereo_yaml.py  kalibr标定目录文件cyperstereo_imu_calibra-results-imucam.txt  -t cyperstereo_sn_c72.yaml -o 新的标定文件
#修改为自己的相机SN编号(相机中间贴纸数字) cyperstereo_sn_c**.yaml
python3 generate_cyperstereo_yaml.py \
  C72-cyperstereo_imu_calibra/cyperstereo_imu_calibra-results-imucam.txt \
  -t cyperstereo_sn_c72.yaml \
  -o cyperstereo_sn_c**.yaml 
