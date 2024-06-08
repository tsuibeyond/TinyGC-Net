import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  

class TinyGC2L_Net(nn.Module):
    def __init__(self, device):
        super(TinyGC2L_Net, self).__init__()
        # parameters' initial
        self.device = device

        # IMU 校正网络
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,3), stride=1)
        self.relu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(1,3), stride=1)
        self.relu2 = nn.PReLU()


        # 创建一个1x3的卷积核，中间元素为1，其余为0  
        self.conv1.weight.data[0] = torch.tensor([[[1., 0., 0.]]], dtype=torch.float32)
        self.conv1.weight.data[1] = torch.tensor([[[0., 1., 0.]]], dtype=torch.float32)
        self.conv1.weight.data[2] = torch.tensor([[[0., 0., 1.]]], dtype=torch.float32)
        self.conv1.bias.data.zero_()

        self.conv2.weight.data[0] = torch.tensor([[[1., 0., 0.]]], dtype=torch.float32)  
        self.conv2.weight.data[1] = torch.tensor([[[0., 1., 0.]]], dtype=torch.float32)  
        self.conv2.weight.data[2] = torch.tensor([[[0., 0., 1.]]], dtype=torch.float32)  
        self.conv2.bias.data.zero_()

    def calibrate_gyro(self, gyro_set):
        gyro_set = torch.unsqueeze(gyro_set, 1)         # 增加一个通道维度 torch.Size([10, 1, 1050, 3])
        # 经过网络
        gyro_set_AA = self.conv1(gyro_set)  # 输入 [1000, 1, 226, 3] | 输出 [1000, 3, 226, 1]
        gyro_set_AA = gyro_set_AA.permute(0,3,2,1)  # 矩阵维度的调整至 [1000, 1, 226, 3]
        gyro_set_AA = self.relu1(gyro_set_AA)

        gyro_set_BB = self.conv2(gyro_set_AA)
        gyro_set_BB = gyro_set_BB.permute(0,3,2,1)
        gyro_set_BB = self.relu2(gyro_set_BB)
        
        gyro_set_cali =  gyro_set_BB + gyro_set

         # 结束网络后，恢复数据维度
        gyro_set_cali = torch.squeeze(gyro_set_cali, 1)         #  [10, 1050, 3]

        return gyro_set_cali


    def forward(self, timestampns_set, gyro_set, start_quat):
        CNT = timestampns_set.shape[1]

        gyro_set_cali = self.calibrate_gyro(gyro_set)
        
        for idx in range(CNT-1): 
            tmp_gyro = (gyro_set_cali[:,idx,:] + gyro_set_cali[:,idx + 1,:])*0.5
            dt_s = (timestampns_set[:,idx+1] - timestampns_set[:,idx]) #
            dt_s = torch.stack([dt_s, dt_s, dt_s],dim=1)
            dtheta_half = tmp_gyro*dt_s*0.5
            ones = torch.ones(dtheta_half.shape[0],1).to(self.device)
            dq = torch.cat([ones, dtheta_half], 1)
            # dq = torch.nn.functional.normalize(dq, p=2, dim=1, eps=1e-12, out=None)
            start_quat = self.quaternion_raw_multiply(start_quat, dq)
            start_quat = torch.nn.functional.normalize(start_quat, p=2, dim=1, eps=1e-12, out=None)
        outcome = start_quat
        return outcome
    
    
    def quaternion_raw_multiply(self, a, b):
        """
        Multiply two quaternions.
        Usual torch rules for broadcasting apply.

        Args:
            a: Quaternions as tensor of shape (..., 4), real part first.
            b: Quaternions as tensor of shape (..., 4), real part first.

        Returns:
            The product of a and b, a tensor of quaternions shape (..., 4).
        """
        aw, ax, ay, az = torch.unbind(a, -1)
        bw, bx, by, bz = torch.unbind(b, -1)
        ow = aw * bw - ax * bx - ay * by - az * bz
        ox = aw * bx + ax * bw + ay * bz - az * by
        oy = aw * by - ax * bz + ay * bw + az * bx
        oz = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack((ow, ox, oy, oz), -1)