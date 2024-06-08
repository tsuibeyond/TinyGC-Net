import os
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

#＃＃ 定义MyDataset类， 继承Dataset, 重写抽象方法：__len()__, __getitem()__
class MyDataset(Dataset):
    
    def __init__(self, dataset_name, category, transform=None): # 举例 data_dir = './data/train'
        currentPath = os.getcwd().replace('\\','/')    # 获取当前路径
        self.project_path = os.path.abspath(os.path.join(currentPath, ".."))
        self.category = category
        self.data_dir = os.path.join(self.project_path, 'data', dataset_name, category)
        self.transform = transform # 单独建一个类进行定义
        self.size = 0 # 在__init__函数的for循环中进行累加
        self.names_list = [] # 在__init__函数的for循环中进行累加
        
        assert os.path.exists(self.data_dir)
        assert os.path.exists(self.project_path)
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                path = os.path.join(root, file)
                self.names_list.append(path)
                self.size += 1

        self.config_path = os.path.join(self.project_path, "configs", "euroc_const_bias.yaml")
        self.config_path_is_exist = os.path.exists(self.config_path)
        # print(self.config_path_is_exist and 11 == self.count_yaml_entries(self.config_path))
        if self.config_path_is_exist and 11 == self.count_yaml_entries(self.config_path):
            with open(self.config_path, 'r') as file:
                self.bias_config = yaml.safe_load(file)  
                # print(f" {self.config_path} is ready")
        # else:  
            # print(f"文件 {self.config_path} is Not ready。")

    def count_yaml_entries(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            if isinstance(data, dict): 
                return len(data)  
            elif isinstance(data, list):  
                return len(data)  
            else:  
                return 0  # 如果 YAML 内容既不是字典也不是列表，则返回 0  

    # 获取样本总数 size
    def __len__(self):
        return self.size
    
    # 
    def __getitem__(self, index):
        file_path = self.names_list[index].split(' ')[0]
        assert os.path.isfile(file_path)

        data = np.load(file_path)
        # 解析数据
        imu_set = np.array(data['arr_0'], dtype='float32')
        # print(imu_set.shape)
        timestampns_set = imu_set[:,0]
        # print('imu_set.shape: {}'.format(imu_set.shape))
        # print('timestamp_set.shape: {}'.format(timestamp_set.shape))
        if self.config_path_is_exist and 11 == self.count_yaml_entries(self.config_path):
            tmp_file_path = file_path.split('/')[-1]
            tmp_file_path = tmp_file_path[tmp_file_path.find('_')+1:tmp_file_path.rfind('_')]
            tmp_file_path = (tmp_file_path[:tmp_file_path.rfind('_')] + tmp_file_path[tmp_file_path.rfind('_')+1:]).lower()
            bias = np.array(self.bias_config[tmp_file_path], dtype=np.float32)
            gyro_set = imu_set[:,1:4] + bias[0:3]
            acc_set = imu_set[:,4:7]
            quat_set = imu_set[:,7:]
            start_quat = quat_set[0]
            end_quat = quat_set[-1]
        else:
            gyro_set = imu_set[:,1:4]
            acc_set = imu_set[:,4:7]
            quat_set = imu_set[:,7:]
            start_quat = quat_set[0]
            end_quat = quat_set[-1]

        sample = {
            'label': torch.from_numpy(end_quat),
            'timestampns_set': torch.from_numpy(timestampns_set),
            'gyro_set': torch.from_numpy(gyro_set),
            'start_quat': torch.from_numpy(start_quat)
        }
        return sample, file_path
        # return sample