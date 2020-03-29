from builtins import enumerate

import cv2
import json
import shutil
import time
from collections import OrderedDict
import platform
from itertools import chain

import requests
import os
import pandas as pd
import numpy as np
# import matplotlib.pylab as plt

# 判断操作系统
import config

OS_TYPE = platform.system()


def load_inlet(condition):
    """
    载入分水情况表
    :return:
    """
    tab = pd.read_csv('./Setting/inlet{}.csv'.format(condition))
    return tab.values[:, 1:]


def react_with_exe(env_path, step=-1):
    """
    与水力学模型交互
    :return:
    """
    if OS_TYPE == 'Linux':
        # 请求windows上的水利学模型服务器进行预算
        # rv = os.system('cd {} && ./HydroDynamic 1>/dev/null 2>&1'.format(env_path))
        rv = os.system('cd {} && ./HydroDynamic'.format(env_path))
        # rv = os.system('cd {} && WINEDEBUG=-all wine Hydrodynamic_model.exe 1>/dev/null 2>&1'.format(env_path))
        # rv = os.system('cd {} && wine Hydrodynamic_model.exe'.format(env_path))
        if str(rv) == '15104':
            raise Exception('\n\n\n水力学模型异常', env_path, step, '\n\n\n')
        return str(rv)
        # rv = requests.get('http://192.168.211.129:5000/hydrodynamic').content
        # return str(rv.decode())
    elif OS_TYPE == 'Windows':
        # 直接调用水力学模型exe
        # rv = os.system('WINEDEBUG=-all wine Hydrodynamic_model.exe')
        rv = os.system('cd {} && Hydrodynamic_model.exe'.format(env_path))
        return str(rv)
    else:
        raise Exception('未知操作系统，无法与水力学模型交互！')


def load_acdict():
    file_path = 'Setting/acdict_{}.json'.format(config.ACTION_MODE)
    action_dict = json.load(open(file_path, 'r'))
    # 转换为数值形式
    action_dict = {int(k): int(v) for k, v in action_dict.items()}
    # 找到静止action 即value为0的action
    mid_action = [action_index for action_index, action_value in action_dict.items()
                  if action_value == 0][0]
    # print('载入了action策略: {}'.format(file_path), 'mid:', action_dict[mid_action])
    return action_dict, mid_action


def getrange(start_gate, ngate):  # 每个闸门闸孔开度的调节范围
    gaterange = OrderedDict()
    filePath = 'Setting/zkkdrange.json'
    zkkddict = json.load(open(filePath, 'r'))
    for i in range(ngate):
        gt = str(int(start_gate) + i)
        gaterange[gt] = zkkddict[gt]
    return gaterange


def getupllrange():  # 上边界闸门流量的范围
    return [51, 189]


def getdownzqswrange(start_gate, ngate):  # 下边界闸门的闸前水位的范围
    filePath = 'Setting/downzqswrange.json'
    zqswrange = json.load(open(filePath, 'r'))
    return zqswrange[str(ngate)]


def getzqswrange(start_gate, ngate):  # 每个闸的最大最小闸孔开度
    minzqsw = []
    maxzqsw = []
    filePath = 'Setting/zqswrange.json'
    zqswdict = json.load(open(filePath, 'r'))
    for i in range(ngate):
        gt = str(int(start_gate) + i)
        maxmin = zqswdict[gt]
        minzqsw.append(maxmin[0])
        maxzqsw.append(maxmin[1])
    return minzqsw, maxzqsw


def getsjsw(start_gate, ngate):
    filePath = 'Setting/sjsw.json'
    sjswdict = json.load(open(filePath, 'r'))
    sjsw = []
    for i in range(ngate):
        gt = str(int(start_gate) + i)
        sjsw.append(sjswdict[gt])
    return sjsw


def getblzkkd(start_gate, ngate):
    filePath = 'Setting/baselinezkkd.json'
    zkkddict = json.load(open(filePath, 'r'))
    zkkd = []
    for i in range(ngate):
        gt = str(int(start_gate) + i)
        zkkd.append(zkkddict[gt])
    return zkkd


def getblzqsw(start_gate, ngate):
    filePath = 'Setting/baselinezqsw.json'
    zqswdict = json.load(open(filePath, 'r'))
    zqsw = []
    for i in range(ngate):
        gt = str(int(start_gate) + i)
        zqsw.append(zqswdict[gt])
    return zqsw


def new_line_adapter(lines):
    """
    # 替换列表中每一行的\n为windows的格式\r\n
    :param lines: 字符串列表
    :return:
    """
    if OS_TYPE == 'Linux':
        for idx, _ in enumerate(lines):
            lines[idx] = lines[idx].replace('\n', '\r\n')
    elif OS_TYPE == 'Windows':
        pass
    else:
        raise Exception('未知操作系统，无法与水力学模型交互！')
    return lines


def generate_result_time_0():
    """
    初始化INPUTFILEALL的RESULT_TIME_0文件
    :return:
    """
    raise NotImplementedError('请确定INPUTFILEALL中的RESULT_TIME_0是空的, 再执行这个初始化操作, 且该操作只需要执行一次')
    root_path = './INPUTFILEALL'
    for input_path in os.listdir(root_path):
        print(input_path)
        prefix_from = os.path.join(root_path, input_path, 'INPUTFILE')
        prefix_to = './INPUTFILE'
        # 复制到INPUTFILES
        for l in os.listdir(prefix_from):
            shutil.copy(os.path.join(prefix_from, l), prefix_to)

        # 运行一次模型 此时模型会有一个警告 因为缺少result_time_0 直接忽略这一信息
        react_with_exe()
        # 将waterdepth复制回INPUTFILES
        copy_waterdepth_to_result_time_0()
        # 把生成的RESULT_TIME_0覆盖回INPUTFILEALL中
        shutil.copy(os.path.join(prefix_to, 'RESULT_TIME_0.txt'), os.path.join(prefix_from), )


def parse_save_log(log_path, start_gate=49, n_gates=6, winlen=4, step=2, vis=False):
    # 闸门数
    state_file_list = os.listdir(os.path.join(log_path, 'output_log'))
    effective_data_right_range = -5
    # 读数据
    # 根据数字排序
    state_file_list.sort(key=lambda f: int(f[:-4]))
    states = []
    data = None
    for state_file in state_file_list:
        data_path = os.path.join(log_path, 'output_log', state_file)
        data = pd.read_csv(data_path)
        if data.empty:
            continue
        states.append(data.values[0][1:effective_data_right_range])

    # 保存原始数据
    states = np.vstack(states)
    pd.DataFrame(states).to_csv(
        os.path.join(log_path, 'all.csv'),
        header=data.columns.values[1:effective_data_right_range]
    )

    # 生成窗口
    windows = [states[window_start:window_start + winlen, :]
               for window_start in range(0, states.shape[0] - winlen - 1, step)]
    # 对每个窗口做索引变换 取出对应的数值
    # 0代表zqsw 3代表ll 6代表每个闸门的总属性数 T代表同一个闸门窗口为行 reshape(-1)是规约到一行
    zqsw_list = np.vstack([window[:, 0::6].T.reshape(-1) for window in windows])
    ll_list = np.stack([window[:, 3::6].T.reshape(-1) for window in windows])
    # 生成header
    header = [["#{}".format(start_gate + idx)] * winlen for idx in range(n_gates)]
    header = list(chain(*header))  # 展平
    # 写入csv
    pd.DataFrame(zqsw_list).to_csv(os.path.join(log_path, 'zqsw.csv'), header=header)
    pd.DataFrame(ll_list).to_csv(os.path.join(log_path, 'll.csv'), header=header)

    if vis:
        # 最后的6是每个闸门属性个数
        states = states.reshape([-1, n_gates, 6])
        # 可视化
        plt.gcf().set_size_inches(12, 4 * 6)
        cate_list = ['zqsw', 'zhsw', 'zkkd', 'll', 'sjsw', 'difference', ]
        for idx, cate in enumerate(cate_list):
            plt.subplot(6, 1, idx + 1)
            for gate_id in range(n_gates):
                plt.plot(states[:, gate_id, idx], label='%d' % (49 + gate_id))
            plt.title(cate)
            plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()


def analyse_log(log_file_path, num_gate, start_gate=48):
    data = pd.read_csv(log_file_path).values
    # 正常闸门数据
    boundary_data = data[:, -1]
    data = data[:, 1:-4]
    # 读取数据
    # zqsw_list = data[:, 0::6]
    # zhsw_list = data[:, 1::6]
    # zkkd_list = data[:, 2::6]
    # ll_list = data[:, 3::6]
    # difference_list = data[:, 5::6]
    title_type_dict = {'zqsw': 0, 'zhsw': 1, 'zkkd': 2, 'zkkd_detail': 2, 'll': 3, 'xz': 4, 'diff': 5, 'diff_detail': 5}
    # 绘图
    for title_type, data_type in title_type_dict.items():
        # 详情绘图
        if title_type in ['zqsw', 'zhsw', 'zkkd_detail', 'll', 'diff_detail']:
            plt.gcf().set_size_inches(12, 2 * num_gate)
            for gate_id in range(num_gate):
                plt.subplot(num_gate, 1, gate_id + 1)
                plt.title('{}-{}'.format(start_gate + gate_id, title_type))
                plt.plot(data[:, data_type::6][:, gate_id])
        # 简略绘图
        elif title_type in ['zkkd', 'diff']:
            plt.gcf().set_size_inches(12, 5)
            for gate_id in range(num_gate):
                plt.plot(data[:, data_type::6][:, gate_id],
                         label='{}'.format(start_gate + gate_id, title_type))
            plt.title('{}'.format(title_type))
            plt.legend(loc='upper left', ncol=3)
        plt.tight_layout()
        plt.show()


def vis_control(num_tests, eps_len, reward_sum):
    water_depth = pd.read_csv('./Pool/PID9998/INPUTFILE/RESULT_TIME_0.txt', sep='\s+', encoding='gbk').values[::-1, :]

    # print(water_depth)
    plt.gcf().set_size_inches(10, 5)
    plt.subplot(2, 1, 1)
    plt.plot(water_depth[:, 0])
    plt.xlim(0, 130)
    plt.ylim(0, 8.5)
    plt.ylabel('Depth')

    # minzqsw, maxzqsw = getzqswrange(48, 5)
    # pos_list = [16, 53, 77, 107, 114]
    # base_level = [70.403, 67.787, 66.721, 65.344, 64.554]
    #
    # for idx, pos in enumerate(pos_list):
    #     plt.plot([pos, pos + 2], [minzqsw[idx] - base_level[idx]] * 2)
    #     plt.plot([pos, pos + 2], [maxzqsw[idx] - base_level[idx]] * 2)

    plt.subplot(2, 1, 2)
    plt.plot(water_depth[:, 1])
    plt.xlim(0, 130)
    plt.ylim(0, 120)
    plt.ylabel('Flow')
    plt.xlabel('{:4d} {:4d} {:5f}'.format(num_tests, eps_len, reward_sum), fontdict={'size': 16, })
    # plt.show()
    plt.savefig('./logs/vis.png')
    plt.clf()
    log_img = cv2.imread('./logs/vis.png')
    cv2.imshow('VIS', log_img)
    cv2.waitKey(10)


def combine_video():
    base_path = './logs/vis'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoWriter = cv2.VideoWriter('saveVideo.avi', fourcc, 6, (1200, 500))
    for idx, img_path in enumerate(sorted(os.listdir(base_path))):
        img = cv2.imread(os.path.join(base_path, img_path))
        cv2.putText(
            img, str(idx + 1), (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX, 3.5,
            (255, 0, 0) if idx < 90 else (0, 0, 255),
            2, cv2.LINE_AA
        )
        videoWriter.write(img)
    videoWriter.release()


def plot_reward():
    root_path = '/media/f214/workspace/sky/logs/2019-06-29_19-44-17_U.sw_D.ll.stable_dis/output_log'
    files = sorted(int(f.replace('.csv', '')) for f in os.listdir(root_path))
    data = []
    for f_id in files:
        reward = pd.read_csv(os.path.join(root_path, '{}.csv'.format(f_id))).values[:, -4]
        reward = np.sum(reward)
        data.append(reward)
        print(f_id, reward)
    plt.gcf().set_size_inches(12, 4)
    plt.plot(data)
    plt.ylabel('R')
    plt.xlabel('Test epoch')
    plt.show()
    pd.DataFrame(data).to_csv('/tmp/625_reward_series.csv')


def gen_inlet_60():
    condition_table = pd.read_csv('需求文档/60个闸稳态工况/data_description/inlet_range.csv').values
    print(condition_table)
    base_line = [inlet[1] for inlet in condition_table]
    headers = ['时间片（11天）'] + [inlet[0] for inlet in condition_table] + ['下游流量']
    inlet_table = np.array([[step] + base_line + [44.13] for step in range(133)])  # 生成inlet时加上时间戳和下游边界
    print(inlet_table, headers)

    # 生成分水表
    pd.DataFrame(inlet_table).to_csv(
        './Setting/inlet.csv', header=headers, index=False
    )

    # 生成配置项
    for condition in condition_table:
        print('[\'{}\', {}],'.format(condition[0], int(condition[3] / condition[1] * 100) if condition[3] != 0 else 0))


def gen_range_60():
    range_table = pd.read_csv('需求文档/60个闸稳态工况/data_description/range.csv')
    gate_ids = list(map(str, range(2, 2 + len(range_table))))

    # 重写设计水位
    sjsw_table = range_table['目标水位'].values
    open('./Setting/sjsw.json', 'w').write(json.dumps(
        dict(list(zip(gate_ids, sjsw_table))), indent=1
    ))

    # 重写闸口开度范围
    zkkd_range_table = map(list, range_table[['开度上限', '开度下限']].values.astype(np.float))
    open('./Setting/zkkdrange.json', 'w').write(json.dumps(
        dict(list(zip(gate_ids, zkkd_range_table))), indent=2
    ))

    # 重写闸前水位范围
    zqsw_range_table = map(list, range_table[['闸前水位上限', '闸前水位下限']].values.astype(np.float))
    open('./Setting/zqswrange.json', 'w').write(json.dumps(
        dict(list(zip(gate_ids, zqsw_range_table))), indent=2
    ))


if __name__ == '__main__':
    # parse_save_log(log_path='/media/f214/workspace/sky/logs/AUDIT/2019-04-12_16-12-42')
    analyse_log(
        log_file_path='logs/2019-11-22_14-07-41_U.sw_D.ll.60gates/output_log/-1.csv',
        num_gate=60, start_gate=2
    )
    # vis_control()
    # combine_video()
    # plot_reward()
    # gen_inlet_60()
    # gen_range_60()
