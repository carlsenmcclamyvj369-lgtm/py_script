import os
import sys
import getopt
import time
import datetime
import numpy as np
import multiprocessing
from multiprocessing import Process, Pool
from PIL import Image
import re


def run_test(test_cmd):
    print(test_cmd)
    os.system(test_cmd)


def get_test_case(test_file):
    fp = open(test_file)
    if fp == False:
        print(test_file + " not avaliable")
        return 1
    test_case_list = list()
    test_case_cmd = list()
    line = fp.readline()
    while line:
        if line.strip().startswith("./run_test"):
            st_res = line.split(' ')
            test_case = st_res[1]
            test_case_list.append(test_case)
            test_case_cmd.append(line)
        line = fp.readline()
    return test_case_list, test_case_cmd


def get_collect_list(folder, tc_list):
    file_name_list = list()
    for test_case in tc_list:
        case_folder = folder + '/' + test_case + "/out1/"
        filenames = os.listdir(case_folder)
        #print(filenames)
        f_names = [fn for fn in filenames if '.png' in fn or '.bmp' in fn]
        #print(f_names)
        if len(f_names) == 0:
            continue
        f_names.sort(key=lambda x: int(re.findall(r'\d+', x)[0]))
        last_idx = re.findall(r'\d+', f_names[-1])[0]
        #print(f_names)
        for f in f_names:
            if last_idx in f:
                file_name_list.append(os.path.join(test_case, 'out1', f))
                print(os.path.join(test_case, 'out1', f))
    return file_name_list


def copy_images(img_prefix, copies=120):
    src_dir = "."
    dst_dir = "."
    img_files = sorted([
        f for f in os.listdir(src_dir)
        if f.startswith(img_prefix) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])
    if not img_files:
        print(f"current folder not find prefix '{img_prefix}' images")
        return
    frame_idx = 0
    for img_name in img_files:
        src_path = os.path.join(src_dir, img_name)
        img = Image.open(src_path).convert("RGB")
        for _ in range(copies):
            dst_name = f"frame_temp{frame_idx:04d}.bmp"
            dst_path = os.path.join(dst_dir, dst_name)
            img.save(dst_path, format="BMP")
            frame_idx += 1
    print(f"finish copy {copies} {img_prefix} images")


def _video_worker(args):
    case_cmd, folder, cur_path = args
    case = case_cmd.replace(' ', '').replace(' ', '')
    st_res = case.split(' ')
    if len(st_res) <= 5 or st_res[5] == '':
        return
    b_video = int(st_res[5])
    if b_video == 0:
        return
    fn = st_res[1].replace('/', '_')
    start_frame = 0
    if b_video == 1:
        os.chdir(os.path.join(cur_path, folder, st_res[1], "out1"))
        mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number " + str(start_frame) + \
        " -i mmr_input%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_in.mp4 >& output.log"
        print(mpeg_enc_cmd)
        os.system(mpeg_enc_cmd)
        mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number " + str(start_frame) + \
        " -i mmr_output%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_in.mp4 >& output.log"
        print(mpeg_enc_cmd)
        os.system(mpeg_enc_cmd)
        cp_cmd = "cp "+fn+"*.mp4 " + os.path.join(cur_path, folder, "cmped_img") + "/"
        os.system(cp_cmd)
        print(cp_cmd)
    elif b_video == 2:
        os.chdir(os.path.join(cur_path, folder, st_res[1], "out1"))
        copy_images("mmr_input")
        os.system("rm -rf frame_temp*.bmp")
        copy_images("mmr_output")
        mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number " + str(start_frame) + \
        " -i frame_temp%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_out.mp4 >& output.log"
        print(mpeg_enc_cmd)
        os.system(mpeg_enc_cmd)
        os.system("rm -rf frame_temp*.bmp")
        cp_cmd = "cp "+fn+"*.mp4 " + os.path.join(cur_path, folder, "cmped_img") + "/"
        os.system(cp_cmd)
        print(cp_cmd)


def create_video(test_case_cmd, folder, process_num=4):
    os.system("mkdir " + folder + "/cmped_img")
    cur_path = os.getcwd()
    args_list = [(cmd, folder, cur_path) for cmd in test_case_cmd]
    pool = Pool(process_num)
    pool.map(_video_worker, args_list)
    pool.close()
    pool.join()


def _cmpedimg_worker(args):
    f, folder_cur, cwd = args
    ln_src = cwd + '/' + folder_cur + '/' + f
    dst_f_name = f.replace('/','#')
    ln_dst = folder_cur + "/cmped_img/"+dst_f_name
    os.system("cp "+ln_src+" "+ln_dst)


def create_cmpedimg_link(file_list, case_list, folder_cur, process_num=4):
    os.system("mkdir " + folder_cur + "/cmped_img")
    cwd = os.getcwd()
    args_list = [(f, folder_cur, cwd) for f in file_list]
    pool = Pool(process_num)
    pool.map(_cmpedimg_worker, args_list)
    pool.close()
    pool.join()


def multi_process_test(folder, test_list_cmd, process_num):
    pool = multiprocessing.Pool(processes = process_num)
    for tc_cmd in test_list_cmd:
        tc_cmd = tc_cmd.strip().replace('\n','').replace('\r','').replace('$1',folder)
        tc_cmd = tc_cmd + ' >> '+folder+'/output.log'
        # print(tc_cmd)
        pool.apply_async(run_test, (tc_cmd,))
        time.sleep(4)
    pool.close()
    pool.join()
    print("multi test done")


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'-h-f:-v',['help','filename=','version'])
    test_case_file = sys.argv[1]
    output_folder = sys.argv[2]
    p_num = int(sys.argv[3])

    diff_file = output_folder + '/diff.txt'
    os.system("mkdir " + output_folder)
    os.system('svn diff /home/zhihong.he/Download/ip_memc_nr_vpss/model_nr > ' + diff_file)
    os.system('cp -rf /home/zhihong.he/Download/ip_memc_nr_vpss/model_nr ' + output_folder)

    tc_list, test_case_cmd = get_test_case(test_case_file)
    multi_process_test(output_folder, test_case_cmd, p_num)

    file_name_list = get_collect_list(output_folder, tc_list)
    create_video(test_case_cmd, output_folder)
    create_cmpedimg_link(file_name_list, tc_list, output_folder)