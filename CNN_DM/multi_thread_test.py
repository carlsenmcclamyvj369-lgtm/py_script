import os
import sys
import getopt
import time
import datetime
import numpy as np
import multiprocessing
from multiprocessing import Process
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
        if line.strip().startswith("./run_test") or line.strip().startswith("python"):
            st_res = line.strip().split()
            test_case = st_res[-1]
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


def create_video(test_case_cmd, folder):
    os.system("mkdir " + folder + "/cmped_img")
    case = ""
    cur_path = os.getcwd()
    for case_cmd in test_case_cmd:
        case = case.replace(' ', '').replace(' ', '')
        st_res = case.split(' ')
        b_video = 0
        if(len(st_res) > 5 and st_res[5] != ''):
            b_video = int(st_res[5])
        if b_video == 1:
            #print(folder, st_res[1])
            fn = st_res[1].replace('/', '_')
            start_frame = int(st_res[3]) + 10
            start_frame = 0
            os.chdir(folder + "/" + st_res[1] + "/out1")
            mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number "+str(start_frame)+\
            " -i mmr_input%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_in.mp4 >& output.log"
            print(mpeg_enc_cmd)
            os.system(mpeg_enc_cmd)
            mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number "+str(start_frame)+\
            " -i mmr_output%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_in.mp4 >& output.log"
            print(mpeg_enc_cmd)
            os.system(mpeg_enc_cmd)
            cp_cmd = "cp "+fn+"*.mp4 "+cur_path+"/"+folder+"/cmped_img/"
            os.system(cp_cmd)
            print(cp_cmd)
            os.chdir(cur_path)
        elif b_video == 2:
            #print(folder, st_res[1])
            fn = st_res[1].replace('/','_')
            start_frame = int(st_res[3]) + 10
            start_frame = 0
            os.chdir(folder + "/" + st_res[1] + "/out1")
            copy_images("mmr_input")
            os.system("rm -rf frame_temp*.bmp")
            copy_images("mmr_output")
            mpeg_enc_cmd = "/export/vls_tools/ffmpeg4.0.2/bin/ffmpeg -framerate 30 -start_number "+str(start_frame)+ \
            " -i frame_temp%4d.bmp -c:v libx265 -pix_fmt yuv420p -x265-params qp=0 -y " + fn + "_out.mp4 >& output.log"
            print(mpeg_enc_cmd)
            os.system(mpeg_enc_cmd)
            os.system("rm -rf frame_temp*.bmp")

            cp_cmd = "cp "+fn+"*.mp4 "+cur_path+"/"+folder+"/cmped_img/"
            os.system(cp_cmd)
            print(cp_cmd)
            os.chdir(cur_path)


def create_cmpedimg_link(file_list, case_list, folder_cur):
    # os.system("mkdir " + folder_cur + "/cmped_img")
    for f in file_list:
        ln_src = os.getcwd() + '/' + folder_cur + '/' + f
        dst_f_name = f.replace('/','#')
        ln_dst = folder_cur + "/cmped_img/"+dst_f_name
        # print("cp "+ln_src+" "+ln_dst)
        os.system("cp "+ln_src+" "+ln_dst)


def multi_process_test(folder, test_list_cmd, process_num):
    pool = multiprocessing.Pool(processes = process_num)
    for tc_cmd in test_list_cmd:
        tc_cmd = tc_cmd.strip().replace('\n','').replace('\r','').replace('$1',folder)
        pool.apply_async(run_test, (tc_cmd,))
        time.sleep(4)
    pool.close()
    pool.join()
    print("multi test done")


if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:],'-h-f:-v',['help','filename=','version'])
    if len(sys.argv) < 4:
        print(f"用法: python {sys.argv[0]} <case_list.txt> <output_folder> <进程数>")
        sys.exit(1)
    test_case_file = sys.argv[1]
    output_folder = sys.argv[2]
    p_num = int(sys.argv[3])

    os.system("mkdir " + output_folder)

    tc_list, test_case_cmd = get_test_case(test_case_file)
    multi_process_test(output_folder, test_case_cmd, p_num)

    file_name_list = get_collect_list(output_folder, tc_list)
    create_video(test_case_cmd, output_folder)
    create_cmpedimg_link(file_name_list, tc_list, output_folder)