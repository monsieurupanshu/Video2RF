import numpy as np
import pickle as pk
import multiprocessing
import time
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd 

range_bin_begin=11#11
range_bin_size=128#128
range_bin_end=range_bin_begin+range_bin_size
dop_exp = 2
dop_base = 10 ** dop_exp
ang_exp=5
ang_base=10**ang_exp
NUM_TX = 3
NUM_RX = 4
CHIRP_LOOPS=128
ADC_SAMPLES=256
gpu_device=torch.device('cpu')
NUM_TRX=NUM_TX * NUM_RX
x_bin_size=128
y_bin_size=128

def reordered_gpu(mat, r_idx_mat):  # (batch, 12, 128, 128)
    # batch, ant, height, width=mat.shape
    return mat[:, :, :, r_idx_mat].permute((0, 3, 4, 1, 2))  # (batch, 128, 128, 12, 128)

def fft_with_window_gpu(mat, axis=-1, shift_flag=False):
    mat_shape = mat.shape
    dim = np.ones_like(mat_shape, dtype=int)
    dim[axis] = mat_shape[axis]
    wd = torch.hamming_window(mat_shape[axis], periodic=False, alpha=0.54, beta=0.46, dtype=torch.float32).reshape( tuple(dim)).to(gpu_device)
    result = torch.fft.fft(mat * wd, dim=axis)
    # result = torch.fft.fft(mat, dim=axis)
    if shift_flag:
        result = torch.fft.fftshift(result, dim=axis)
    return result

def stft_with_window_gpu(mat, window_lenth,  hop_length, shift_flag=True):
    wd = torch.hamming_window(window_lenth, periodic=False, alpha=0.54, beta=0.46, dtype=torch.float32).to(gpu_device)
    time_doppler = torch.stft(mat, window_lenth, hop_length, window=wd, center=False)
    if shift_flag:
        time_doppler = torch.fft.fftshift(time_doppler, dim=1)
    return time_doppler


def clutter_removal_gpu(mat, axis=-2): # axis on the chirp dimension
    return mat-mat.mean(dim=axis, keepdims=True)

def normalize_base_exp(mat, base, exp):
    mat=torch.log10(mat-mat.min()+base)-exp
    return mat/mat.max()

def preprocess_batch_length_data(data, steer_mat, r_idx_mat):
    batch_size, length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape
    data = data.view(batch_size * length_size, tx_size, rx_size, chirp_loops, adc_samples)
    with torch.no_grad():
        data = data.cfloat().to(gpu_device)
        frame_size=data.shape[0]
        frames_tensor=data

        batch_length = frames_tensor.shape[0]
        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)#为true的话，0就在中间了
        frames_range = clutter_removal_gpu(frames_range, axis=-2)#不用的话，zero-velocity上会有很多值

        # dop
        channel_doppler = fft_with_window_gpu(frames_range[..., range_bin_begin:range_bin_end], axis=-2, shift_flag=True)
        channel_doppler = torch.abs(torch.mean(channel_doppler, axis=(1, 2)))  # (batch_length, 128, 256) #对12张能量图取平均
        channel_doppler = normalize_base_exp(channel_doppler, dop_base, dop_exp)

        channel_angle = frames_range.reshape(batch_length, NUM_TX * NUM_RX, CHIRP_LOOPS, ADC_SAMPLES)  # (batch_length, 8, 128, 256)
        channel_angle = reordered_gpu(channel_angle, r_idx_mat)
        channel_angle = cal_X_Y_plane_MVDR1D_energy_MP_batch_gpu(channel_angle, steer_mat)  # (batch_length, 128, 256)

        channel_angle = normalize_base_exp(channel_angle, ang_base, ang_exp)
        return channel_doppler, channel_angle

def get_time_doppler(data):
    batch_size, length_size, tx_size, rx_size, chirp_loops, adc_samples = data.shape
    data = data.view(batch_size * length_size, tx_size, rx_size, chirp_loops, adc_samples)
    data=torch.permute(data,(1,2,0,3,4)).view(tx_size*rx_size,batch_size * length_size,chirp_loops, adc_samples)
    with torch.no_grad():
        data = data.cfloat().to(gpu_device)
        frames_tensor = data

        frames_range = fft_with_window_gpu(frames_tensor, axis=-1, shift_flag=False)  # 为true的话，0就在中间了
        frames_range = clutter_removal_gpu(frames_range, axis=-2)  # 不用的话，zero-velocity上会有很多值

        # b*tx*rx*chrip*adc
        # find max along adc, and get 128(chirp)*1 vector
        max_index=torch.argmax(torch.abs(frames_range), axis=-1).numpy()
        index=[]
        for i in range(max_index.shape[0]):
            for j in range(max_index.shape[1]):
                for z in range(max_index.shape[2]):
                    index.append([i,j,z, max_index[i,j,z]])
        index=np.stack(np.array(index))

        # range_max = frames_range[index[:,0],index[:,1], index[:,2]].view(batch_size * length_size*2, chirp_loops//2)
        # time_doppler = fft_with_window_gpu(range_max,shift_flag=True)
        # time_doppler=torch.abs(time_doppler)
        # time_doppler = normalize_base_exp(time_doppler, ang_base, ang_exp).transpose(1,0)

        range_max = frames_range[index[:,0],index[:,1], index[:,2], index[:,3]].view(tx_size*rx_size, batch_size * length_size*chirp_loops)
        time_doppler=stft_with_window_gpu(range_max, 
                                        # window_lenth=64, 
                                        window_lenth=128, 
                                        hop_length=128, shift_flag=True)
        time_doppler=torch.abs(torch.mean(time_doppler,dim=0))
        time_doppler = normalize_base_exp(time_doppler, ang_base, ang_exp)
        return time_doppler

def _init_steer_mat_X_Y_bin_version_gpu(antenna_size, x_bin_size, y_bin_size, y_bin_begin, bin_len): # one bin represents 4 cm for both X and Y, 4.3 for R, y_bin_begin start from 0, bin_len unit: cm
    x_vec=(np.expand_dims(np.arange(x_bin_size-1, -1, -1, dtype=np.float64), axis=1)-(x_bin_size)//2)*bin_len
    y_vec=np.expand_dims(np.arange(y_bin_begin, y_bin_size+y_bin_begin, dtype=np.float64), axis=0)*bin_len
    r_mat=np.sqrt(np.square(x_vec)+np.square(y_vec))
    r_idx_mat=(r_mat/4.3).astype(np.int64) #(x_bin_size, y_bin_size)
    sin_mat=x_vec/r_mat
    #sin_mat=np.clip(sin_mat, -1.0, 1.0)
    steer_mat=[]
    for i in range(antenna_size):
        if i<8:
            steer_mat.append(np.exp(-1j*i*np.pi*sin_mat))
        else:
            steer_mat.append(np.exp(-1j*(i-6)*np.pi*sin_mat))
    steer_mat=np.stack(steer_mat, axis=0) #(antenna_size,x_bin_size,y_bin_size)
    steer_mat=steer_mat.transpose((1,2,0))
    return torch.from_numpy(steer_mat).cfloat().to(gpu_device), torch.from_numpy(r_idx_mat).long().to(gpu_device)

def cal_X_Y_plane_MVDR1D_energy_MP_batch_gpu(rfft, steer_mat): #(batch, 128, 128, 12, 128), (128, 128, 12)
    x_mat=rfft #(batch, 128, 128, 12, 128)
    xH_mat=torch.conj(rfft.permute(0,1,2,4,3)) #(batch, 128, 128, 128, 8)
    R_mat=torch.matmul(x_mat, xH_mat) #(batch,128,128,12,12)
    R_mat=R_mat.cfloat()
    R_1_mat=torch.linalg.pinv(R_mat, hermitian=True) #(batch,128,128,12,12)
    R_1_mat=R_1_mat.cfloat()
    a_mat=steer_mat[None,:,:,:,None] #(1,128,128,12,1)
    aH_mat=torch.conj(steer_mat)[None,:,:,None,:] #(1,128,128,1,12)
    p_mat=1/torch.matmul(torch.matmul(aH_mat, R_1_mat), a_mat)
    data=torch.abs(p_mat.squeeze(-1).squeeze(-1))
    return data #(batch, 128, 128)

if __name__=='__main__':
    # frame_size = 1500
    trial_list=['STFTresult']  ### output directory ###
    # root_loc = './movement_HS3/' ### input directory ###
    begin=0
    end=599
    steer_mat, r_idx_mat = _init_steer_mat_X_Y_bin_version_gpu(antenna_size=NUM_TRX,x_bin_size=x_bin_size,y_bin_size=y_bin_size,y_bin_begin=range_bin_begin,bin_len=4.0)
    #since the data is large, use multiprocessing or different chunk
    path = "./STFTresult/"    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    f = open(r'binlist.txt', "r")
    a = list(f)
    f.close()
    for trial in trial_list:
        for i in range(len(a)): # TEMPORARY MEASUREMENT TO REDUCE THE RAM
            a[i] = a[i].replace('\n', '')
            root_loc = './movement_%s/' % (a[i].replace('.bin', ''))
            inloc =  root_loc

            output_dir = path+a[i].replace('.bin', '')
            outloc = './%s/' % (output_dir)
            if not os.path.isdir(outloc):
                os.mkdir(outloc)            
            # os.system('mkdir %s' % (outloc))
            data=[np.load(inloc+'%04d.npy'%(i)) for i in range(begin, end)]
            data_tensor=torch.from_numpy(np.asarray(data)[None])#1*15*3*4*128*256

            time_doppler = get_time_doppler(data_tensor)
            time_doppler=time_doppler.cpu().numpy()
            plt.matshow(time_doppler,vmin=-1,vmax=1)
            plt.savefig(outloc+'real_doppler.png')
            # plt.savefig(outloc+'range_v{}.jpg'.format(i))
            # plt.show()
            plt.close()
            
            for j in range(568):
                temp = time_doppler[48:80, j:j+32] # we take the 32 rows in the middle of the whole plot, and take 32 columes for each cut
                plt.matshow(temp,vmin=-1,vmax=1)
                plt.savefig(outloc+'real_doppler_dat_%03d.png' % (j))
                plt.close()
                pd.DataFrame(temp).to_csv(outloc+"real_doppler_dat_%03d.csv" % (j))
            
            print('%s done' % (a[i]))

            # channel_doppler, channel_angle=preprocess_batch_length_data(data_tensor, steer_mat, r_idx_mat)
            # for hm in channel_angle:
            #     hm=hm.cpu().numpy()
            #     plt.matshow(hm,vmin=-1,vmax=1)
            #     plt.show()
    """
    for trial in trial_list:
        root_loc = './movement_squat1/'
        inloc =  root_loc
        outloc = './%s/' % (trial)
        os.system('mkdir %s' % (outloc))
        data=[np.load(inloc+'%04d.npy'%(i)) for i in range(begin, end)]
        data_tensor=torch.from_numpy(np.asarray(data)[None])#1*15*3*4*128*256

        # fig = plt.figure()
        time_doppler = get_time_doppler(data_tensor)
        time_doppler=time_doppler.cpu().numpy()
        plt.matshow(time_doppler[48:80, 0:576],vmin=-1,vmax=1)
        plt.show()
        # plt.savefig(path+'range_v_squat1.jpg')
        print('sub1 done')
        '''
        for i in range(18):
            temp = time_doppler[16:48, 32*i:32*i+32] # we take the 32 rows in the middle of the whole plot, and take 32 columes for each cut
            plt.matshow(temp,vmin=-1,vmax=1)
            plt.savefig(path+'range_v_squat1_%03d.jpg' % (i))
            pd.DataFrame(temp).to_csv(path+"velocity_time_plot{}.csv".format(i))
        '''
        csv_path = './0403synthesizeddata/Squats/Squats1/CSV/'
        for j in range(18):
            if j == 0:
                synt = np.genfromtxt(csv_path+'synth_doppler_dat_%03d.csv'%(j), delimiter=',')
            else:
                synt_temp = np.genfromtxt(csv_path+'synth_doppler_dat_%03d.csv'%(j), delimiter=',')
                synt = np.append(synt, synt_temp, axis=1)
        plt.matshow(synt)
        plt.show()
    """