import numpy as np
import pickle as pk
import multiprocessing
import time
import os


class radarConfig():
    def __init__(self):
        self.NUM_TX = 3
        self.NUM_TX_LOWER = 2
        self.NUM_RX = 4
        self.NUM_TRX = self.NUM_TX * self.NUM_RX
        self.CHIRP_LOOPS = 128
        self.ADC_SAMPLES = 256
        self.NUM_RANGE_BINS = self.ADC_SAMPLES
        self.NUM_DOPPLER_BINS = self.CHIRP_LOOPS

        self.FRAME_SIZE = self.NUM_TX * self.NUM_RX * self.CHIRP_LOOPS * self.ADC_SAMPLES

        self.FRAME_BYTE_SIZE = self.FRAME_SIZE * 4
        self.FRAME_INT16_SIZE = self.FRAME_SIZE * 2
        self.FRAME_COMPLEX_SIZE = self.FRAME_SIZE

        self.ANGLE_BIN_SIZE = 128
        self.FPS = 10


class transform():
    def __init__(self):
        self.bin_location = './'
        self.out_root_loc = './'
        self.filename_loc = './'
        self.filename_list = self.load_filenames_by_file(self.filename_loc + 'names.sync')
        self.skip_sec_from_begin = 10
        self.sync_dict = self._load_sync_dict('./frame.sync')
        self.cfg = radarConfig()

    def load_filenames_by_file(self, filename):
        l = []
        with open(filename, 'r') as infile:
            for line in infile:
                if line.strip() == '':
                    continue
                l.append(line.strip())
        return l

    def save_np_mat(self, mat, name):
        np.save(name, mat)

    def _load_sync_dict(self, infilename):
        d = {}
        with open(infilename) as infile:
            for line in infile:
                if line.strip() == '':
                    continue
                seg = line.strip().split(':')
                d[seg[0].strip()] = int(seg[1].strip())
        return d

    def load_binary_frames(self, infilename, batch_length, skip_size):
        frame_byte_size = self.cfg.FRAME_BYTE_SIZE
        frame_int16_size = self.cfg.FRAME_INT16_SIZE
        frame_complex_size = self.cfg.FRAME_COMPLEX_SIZE
        with open(infilename, 'rb') as infile_bin:
            infile_bin.seek(skip_size * frame_byte_size, 0)
            frame_read = np.frombuffer(infile_bin.read(batch_length * frame_byte_size), dtype=np.int16)#读取多少字节，按每两个字节当成int16的方式读取出来
        assert len(frame_read) == frame_int16_size * batch_length
        frame_out = np.zeros(shape=(batch_length * frame_complex_size,), dtype=np.complex_)
        frame_out[0::2] = frame_read[0::4] + 1j * frame_read[2::4]
        frame_out[1::2] = frame_read[1::4] + 1j * frame_read[3::4]
        frame_np = frame_out.reshape(batch_length, self.cfg.CHIRP_LOOPS, self.cfg.NUM_TX, self.cfg.NUM_RX,
                                     self.cfg.ADC_SAMPLES).transpose(0, 2, 3, 1, 4)  # (batch_length, 3, 4, 128, 256)
        return frame_np

    def get_processed_frames(self, infilename, frame_begin, frame_size, skip_size):
        for frame_no in range(frame_size):
            frames_np = self.load_binary_frames(infilename, 1, skip_size + frame_no).squeeze(0)  # (batch,3,4,128,256)
            with open(self.dump_folder + '%04d.npy' % (frame_begin + frame_no), 'wb') as outfile:
                np.save(outfile, frames_np.astype(np.csingle))

    def sprocessing_dump_sample(self, filename_idx, frame_begin, frame_size):
        infilename = self.bin_location + self.filename_list[filename_idx]
        skip_size = self.sync_dict[str(filename_idx)] + self.skip_sec_from_begin * self.cfg.FPS + frame_begin
        print('\t\t', frame_begin, infilename, skip_size)

        self.get_processed_frames(infilename, frame_begin, frame_size, skip_size)
        # q.put({tid: [dop, ang]})
        print('\t\tprocess %d + %d done' % (frame_begin, frame_size))

    def mprocessing_simulated_signal(self, filename_idx, total_length, process_size):
        self.keyword = 'test_data_%02d' % (filename_idx)
        self.dump_folder = self.out_root_loc + '%s/' % (self.keyword)
        if not os.path.isdir(self.dump_folder):
            os.system('mkdir %s' % (self.dump_folder))

        print('\tmultiprocessing begin')
        begin_time = time.time()
        process_list = []
        piece_size = total_length // process_size
        if total_length % process_size != 0:
            process_size += 1
        print('\t\tprocess_size:', process_size)
        for i in range(process_size - 1):
            t = multiprocessing.Process(target=self.sprocessing_dump_sample,
                                        args=(filename_idx, i * piece_size, piece_size))
            t.start()
            process_list.append(t)
        i = process_size - 1
        t = multiprocessing.Process(target=self.sprocessing_dump_sample,
                                    args=(filename_idx, i * piece_size, total_length - piece_size * i))
        t.start()
        process_list.append(t)

        for t in process_list:
            t.join()
        end_time = time.time()
        print('\tmultiprocessing end: %.1f min(s)' % ((end_time - begin_time) / 60))


if __name__ == '__main__':
    t = transform()
    filename_idx_list = list(range(0, 1))
    for filename_idx in filename_idx_list:
        print(filename_idx)
        t.mprocessing_simulated_signal(filename_idx, 1500, 1)
