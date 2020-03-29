# encoding=utf-8
import os
import pdb
import random
import time

import numpy as np
from collections import OrderedDict
import shutil

import config
import utils
import requests
import platform

# 判断操作系统
OS_TYPE = platform.system()


class WaterModel(object):
    def __init__(self, gates, start_gate, env_id):
        targetLevel = utils.getsjsw(start_gate, gates + 1)
        self.targetTable = {str(start_gate + i): targetLevel[i] for i in range(gates + 1)}

        self.rewardTable = config.reward_table

        self.init_base_ll = None  # 上游边界随机浮动基准
        self.init_base_up_zqsw = None  # 上游边界随机浮动基准
        self.init_base_down_zqsw = None  # 下游边界随机浮动基准

        # 载入action dict
        self.acdict, self.mid = utils.load_acdict()  # 从配置文件中加载动作
        assert self.acdict[self.mid] == 0, '载入action错误!!!'

        self.n_actions = len(self.acdict)  # 根据字典来

        # 载入对应工况的INPUTFILE类型
        self.working_condition = config.WORKING_CONDITION

        # 载入分水表
        self.inlet_table = utils.load_inlet(self.working_condition)
        self.cur_step = -1

        # 初始化时给定闸门数和特征数
        self.n_gates = gates
        self.n_features = gates * 4 + 2 + config.INLET_FORWARD_LENGTH * len(self.inlet_table[0, :])
        # 第一个闸门的编号
        self.start_gate = start_gate

        # 上一次的闸前水位记录
        self.lastzqsw = []

        # 闸门24小时的闸前水位记录
        self.zqswlist = []

        for i in range(gates + 1):
            self.zqswlist.append([])

        # 闸门48小时调控与否的记录，调控为1
        # self.changeornot = [0.0] * 24 * gates

        # 多线程时的环境id
        self.env_id = env_id
        self.pid_prefix = './Pool/PID{:04}/'.format(env_id)
        self._init_pool_path()

    def _init_pool_path(self):
        """
        初始化进程运行环境
        :return: None
        """
        dll_root = os.path.abspath('./lib')

        if os.path.exists(self.pid_prefix):
            shutil.rmtree(self.pid_prefix)

        os.mkdir(self.pid_prefix)
        os.mkdir(os.path.join(self.pid_prefix, 'OUTPUTFILE'))
        shutil.copytree(
            os.path.join('INPUTFILEALL',
                         '#{}-#{}'.format(self.start_gate, self.start_gate + self.n_gates - 1),
                         # 载入不同类型的inputfile
                         'INPUTFILE{}'.format(self.working_condition)),
            os.path.join(self.pid_prefix, 'INPUTFILE')
        )
        shutil.copy('./lib/HydroDynamic', self.pid_prefix)
        # # shutil.copy('./lib/Hydrodynamic_model.exe', self.pid_prefix)
        # # 链接dll
        # [shutil.copy(os.path.join(dll_root, dll), os.path.join(self.pid_prefix, dll)) for dll in
        #  ['libifcoremdd.dll', 'libmmd.dll', 'libmmdd.dll', 'msvcr100d.dll']]
        # 初始化输入输出路径
        self.INPUTFILE_PATH = os.path.join(self.pid_prefix, 'INPUTFILE')
        self.OUTPUTFILE_PATH = os.path.join(self.pid_prefix, 'OUTPUTFILE')
        # print('载入了INPUTFILE{}'.format(self.working_condition))

    def get_init_action(self):
        initactionlist = []
        for i in range(self.n_gates):
            initactionlist.append(self.mid)
        return initactionlist

    def reset(self, start_gate, n_gate):
        """
        重置模型状态 注意这里要运行一次模型
        :param start_gate: 初始闸门
        :param n_gate: 闸门数
        :return:
        """
        # # 随机选择一种工况
        # self.working_condition = random.choice([
        #     config.INPUT_BOUNDARY_UP,
        #     config.INPUT_BOUNDARY_DOWN
        # ])
        #
        # # 重载inlet
        # self.inlet_table = utils.load_inlet(self.working_condition)

        # 重置INPUTFILE等文件
        self._init_pool_path()

        # 写入初始com element
        self._set_com_element()

        # 运行一次模型 此时模型会有一个警告 因为缺少result_time_0 直接忽略这一信息
        utils.react_with_exe(self.pid_prefix)

        # 复制waterdepth
        self.copy_waterdepth_to_result_time_0()

        # 上一次的闸前水位记录
        self.lastzqsw = []

        # 闸门24小时的闸前水位记录
        self.zqswlist = []
        for i in range(self.n_gates + 1):
            self.zqswlist.append([])

        # 初始化初始基准随机浮动水位状态
        self.init_base_ll = None

        # 转换STATE到1
        self.change_STATE_file(0)

        # 初始化初始分水表
        self.cur_inlet = None

        # 闸门48小时调控与否的记录，调控为1
        # self.changeornot = [0.0] * 24 * gates

    def _set_com_element(self):
        """
        初始化com-element的分水初值
        """
        # 读取com-element
        com_element_file = os.path.join(self.INPUTFILE_PATH, 'COM-ELEMENT.txt')
        with open(com_element_file, 'r', encoding='GBK') as f:
            com_element = [lines for lines in f]

        # 初始分水项
        init_inlet = self.inlet_table[0, :-1]
        # 写入com element的每一项
        for inlet_id, inlet_value in enumerate(init_inlet):
            com_element_inlet = com_element[len(com_element) - len(init_inlet) + inlet_id].split()
            com_element_inlet[-1] = str(inlet_value)
            com_element[len(com_element) - len(init_inlet) + inlet_id] = '\t'.join(com_element_inlet) + '\n'

        # 写回com-element
        # 写回
        with open(com_element_file, 'w', encoding='GBK') as f:
            com_element = utils.new_line_adapter(com_element)  # 转换为windows下的换行
            f.writelines(com_element)

    def getNextState(self, state, action, episode, is_training=False):
        '''
        :param action: 采取的决策
        :function: 根据强化学习给出的state和action
                    1、构造调用exe需要的数据
                    2、调用exe，得到nextState
                    3、根据结果计算reward
        :param state: 字典,闸门-7200时刻和0时刻的闸孔开度
        :return: 字典
        '''

        # 统计48小时内闸门调控次数，先pop掉48小时前的数据，再append本次的调控与否的数据
        # for i in range(self.n_gates): 有bug
        #     self.changeornot.pop(0)
        #     if action[i] != self.mid:
        #         self.changeornot.append(1)
        #     else:
        #         self.changeornot.append(0)

        # 1、构造调用exe需要的数据
        message = self.prepareMessage(state, action)

        # if is_training:
        #     # 限制闸口开度 让其差异不至过大
        #     message = self.constrain_zkkd_diff(message)

        # 2、调用exe，得到nextState(字典)和observation_(np.array)
        next_state, observation_ = self.runExe(message, is_training)

        # 更新变量
        # 注意这里多加了一个下游边界闸门的水位
        for i in range(self.n_gates + 1):
            if len(self.zqswlist[i]) >= 12:
                self.zqswlist[i].pop(0)
            gtnum = str(self.start_gate + i)
            self.zqswlist[i].append(state[gtnum + 'zqsw'][0])
        # 更新变量lastzqsw
        self.lastzqsw = []
        # 注意这里多加了一个下游边界闸门的水位
        for i in range(self.n_gates + 1):
            gtnum = str(self.start_gate + i)
            self.lastzqsw.append(state[gtnum + 'zqsw'][0])

        # 3、根据结果计算reward
        reward, endState = self.getRewardandState(next_state, episode)

        # 本时刻的7200的数据用于构造下一次的getNextState的输入，是下一次的0时刻数据
        newState = OrderedDict()
        for i in range(self.n_gates):
            gtnum = str(self.start_gate + i)
            newState[gtnum + 'zqsw'] = [next_state[gtnum + 'zqsw']]
            newState[gtnum + 'zkkd'] = [message[gtnum + 'zkkd'][1], next_state[gtnum + 'zkkd'][0]]
        # 下游边界闸门
        newState[str(self.start_gate + self.n_gates) + 'zqsw'] = \
            [next_state[str(self.start_gate + self.n_gates) + 'zqsw']]

        # TODO 这里送进网络的观测量 是未加扰动的inlet 即原始inlet表中的inlet 这里需要改进
        # TODO 可以将当前时刻加了扰动的inlet和未来几个时间片未加扰动的合并
        # 将分水表加入观测值中
        observation_ = self._concat_observation_inlet(observation_)

        return observation_, reward, endState, action, newState, next_state

    def prepareMessage(self, state, action):
        """
        :param state: 字典； 'zqsw': 0时刻的闸前水位，仅用于判断action是否合理
                            'zkkd':-7200和0的闸孔开度
        :param action: 对闸门的调控动作
        :return:  message{50zkkd, 51zkkd, 52zkkd}
        """

        message = OrderedDict()
        gaterange = utils.getrange(self.start_gate, self.n_gates)

        for i in range(self.n_gates):
            gtnum = str(self.start_gate + i)
            # if state[gtnum + 'zqsw'][0] < self.targetTable[gtnum] and action[i] > self.mid:
            #     action[i] = self.mid  # 如果当前水位小于目标水位且当前执行的动作的和常识相反，则什么也不做
            # elif state[gtnum + 'zqsw'][0] > self.targetTable[gtnum] and action[i] < self.mid:
            #     action[i] = self.mid  # 如果当前水位大于目标水位且当前执行的动作的和常识相反，则什么也不做

            message[gtnum + 'zkkd'] = np.array([state[gtnum + 'zkkd'][1] + self.acdict[action[i]],
                                                state[gtnum + 'zkkd'][1] + self.acdict[action[i]],
                                                state[gtnum + 'zkkd'][1] + self.acdict[action[i]]])
            message[gtnum + 'zkkd'][message[gtnum + 'zkkd'][:] < gaterange[gtnum][0]] = gaterange[gtnum][0]  # 边界判断
            message[gtnum + 'zkkd'][message[gtnum + 'zkkd'][:] > gaterange[gtnum][1]] = gaterange[gtnum][1]

        return message

    def boundary_ll_disturbance(self, ll):
        """
        边界流量随机扰动
        :param ll: 边界流量
        :param mode: 扰动模式 上 下 稳定
        :return:
        """
        # # 如果没有初始化过随机浮动基准 则进行初始化 否则保持不变
        # self.init_base_ll = ll if self.init_base_ll is None else self.init_base_ll
        # 加随机扰动
        # 直接在初始流量上加扰动 噪声正态分布
        noise = np.random.normal(loc=0.0, scale=0.3, size=None)
        # 限制在正负2之间
        noise = max(-2, min(noise, 2))
        ll = ll + noise
        # # 保证加的随机值在基准点+-10范围内
        # while ll > self.init_base_upll + 5 or ll < self.init_base_upll - 5:
        #     ll = ll + random.uniform(
        #         self.upll_stable_range[0],
        #         self.upll_stable_range[1],
        #     )

        # 截断精度
        # print('INIT: ', self.init_base_upll, ' UPLL: ', upll)
        ll = np.round(ll, 2)
        return ll

    def boundary_level_disturbance(self, up_zqsw, down_zqsw, ):
        """
        边界水位随机扰动
        :param up_zqsw: 上游边界水位
        :param down_zqsw: 下游边界水位
        :return:
        """
        # 初始化基准上下游闸前水位
        self.init_base_up_zqsw = up_zqsw if self.init_base_up_zqsw is None else self.init_base_up_zqsw
        self.init_base_down_zqsw = down_zqsw if self.init_base_down_zqsw is None else self.init_base_down_zqsw
        # 加随机扰动
        up_zqsw = self.init_base_up_zqsw + random.uniform(
            -0.02, 0.02
        )
        down_zqsw = self.init_base_down_zqsw + random.uniform(
            -0.02, 0.02
        )
        # 截断精度
        # print('INIT: ', self.init_base_up_zqsw, ' UP_ZQSW: ', up_zqsw, ' DOWN_ZQSW: ', down_zqsw)
        up_zqsw = np.round(up_zqsw, 2)
        down_zqsw = np.round(down_zqsw, 2)
        return up_zqsw, down_zqsw

    def runExe(self, message, is_training):
        '''
        :function: 根据强化学习给出的动作，配置input，运行exe，返回exe运行结果
        :param message: 字典
        {
         'zkkd':list[list] 行：3(时刻)；列：1（孔数）平均闸孔开度
        }
        :return: 字典
        '''
        # ================================================================================
        # 读写BOUNDARY
        boundary_file = os.path.join(self.INPUTFILE_PATH, 'BOUNDARY.txt')
        with open(boundary_file, 'r', encoding='GBK') as f:
            boundary = [lines for lines in f]

        # 读取boundary的数据
        # 计算开始时间、结束时间、步长
        boundary_line_time_step = boundary[2].strip('\n').split('\t')
        # 上游、下游边界过程第一行
        boundary_line_up_down_process_1 = boundary[6].strip('\n').split('\t')
        # 上游、下游边界过程第二行
        boundary_line_up_down_process_2 = boundary[7].strip('\n').split('\t')

        # 获取上游边界流量和下游边界水位
        up_zqsw = float(boundary_line_up_down_process_2[0])
        down_ll = float(boundary_line_up_down_process_2[1])

        # 上下游边界过程中
        # 旧的第二行直接替换第一行
        boundary[6] = boundary[7]

        # 生成新的第二行数据
        # 对边界水位有一定随机
        # 对上下游水位进行扰动
        if is_training:
            # 上游水位波动
            up_zqsw, _ = self.boundary_level_disturbance(up_zqsw, 0)
            # 下游流量从分水表中读取
            down_ll = self.boundary_ll_disturbance(self.inlet_table[self.cur_step][-1])
            # down_ll = self.inlet_table[self.cur_step][-1]

        boundary[7] = str(up_zqsw) + '\t' + str(down_ll) + '\n'

        # 写回
        with open(boundary_file, 'w', encoding='GBK') as f:
            boundary = utils.new_line_adapter(boundary)  # 转换为windows下的换行
            f.writelines(boundary)

        # ================================================================================
        # 读写INNERBOUNDAY
        # 处理INNERBOUNDAY
        inner_boundary_file = os.path.join(self.INPUTFILE_PATH, 'INNERBOUNDARY.txt')
        with open(inner_boundary_file, 'r', encoding='GBK') as f:
            inner_boundary = [lines for lines in f]

        # 构造下一时刻的zkkd
        new_zkkd_line = []
        for i in range(self.n_gates):
            gtnum = str(self.start_gate + i)
            # 因为message里三个闸口开度都一样 这里直接用第0个代替
            new_zkkd_line.append(str(int(message[gtnum + 'zkkd'][0])))
        new_zkkd_line = '\t'.join(new_zkkd_line) + '\n'
        # zkkd第二行直接覆盖第一行
        inner_boundary[2] = inner_boundary[3]
        # 第二行是新的闸口开度
        inner_boundary[3] = new_zkkd_line

        # 分退水按照分退水表执行
        if is_training:
            inner_boundary[7] = inner_boundary[8]
            inner_boundary[8] = self._build_inlet_varience()
        else:
            # 如果是预热过程 每次都从inlet第一行赋值给innerboundary
            assert self.cur_step <= 0, '请检查当前是否为预热过程!!!'
            inner_boundary[7] = self._build_init_inlet()
            inner_boundary[8] = self._build_init_inlet()

        # 写回
        with open(inner_boundary_file, 'w', encoding='GBK') as f:
            inner_boundary = utils.new_line_adapter(inner_boundary)  # 转换为windows下的换行
            f.writelines(inner_boundary)

        rv = utils.react_with_exe(self.pid_prefix, self.cur_step)
        if rv != '0':
            # raise Exception('**** 水力学模型异常: ')
            print('**** 水力学模型异常: ', rv)

        dynamic_results = np.loadtxt(os.path.join(self.OUTPUTFILE_PATH, 'DynamicRESULT.txt'))

        # 最后一个时刻对应的行数 可能是0时刻 也可能是2时刻
        last_axis = 4
        next_state = OrderedDict()
        # 这一项实际上为第一个闸门的流量　和后边的观测量重复了
        next_state[str(self.start_gate) + 'll_2'] = dynamic_results[last_axis, 1]
        # 输入是下游流量　输出是下游水位
        next_state[str(self.start_gate + self.n_gates) + 'zqsw'] = dynamic_results[last_axis, self.n_gates * 3 + 2]
        for i in range(self.n_gates):
            gtnum = str(self.start_gate + i)
            next_state[gtnum + 'zqsw'] = dynamic_results[last_axis, i * 3 + 2]
            next_state[gtnum + 'zhsw'] = dynamic_results[last_axis, i * 3 + 3]
            next_state[gtnum + 'll'] = dynamic_results[last_axis, i * 3 + 4]
            # 7200时刻的，下一次输入将被当作0时刻
            next_state[gtnum + 'zkkd'] = np.reshape(message[gtnum + 'zkkd'][0].astype(float), [-1])

        # 把字典转换成list
        observation_ = []
        observation = list(next_state.values())
        for o in observation:
            if isinstance(o, float) is True:
                observation_.append(o)
            else:
                observation_.extend(o.tolist())

        # 保存边界条件
        next_state['up_zqsw'] = up_zqsw
        next_state['down_ll'] = down_ll
        next_state['inlet'] = self.cur_inlet if self.cur_inlet else []

        return next_state, np.array(observation_)

    def getRewardandState(self, resultMessage, episode):
        """
        EndState = 0: 调整后的水位差值在0-5之间
        EndState = 1: 调整后的水位差值在5-20之间
        EndState = 2XX: 调整后XX闸的水位差值在20以上，这是崩了的情况
        EndState = 3: 调整后出现部分值超出水动力模型范围，出现NAN情况
        EndState = 4XX: XX闸前水位2小时水位变幅超过15cm
        EndState = 5XX: XX闸前水位24小时水位变幅超过30cm
        EndState = 6XX: XX闸前水位超出约束
        :param resultMessage: 水动力模型的输出
        :return:
        """
        # 控制添加endstate为3~6的约束条件
        stateflag3 = True
        stateflag4 = True
        stateflag5 = True
        stateflag6 = True

        endState = ''
        # 奖励列表
        reward_list = []

        # episode 的区间分值控制
        if episode < 30:
            end3thresh = 0.5  #
            end2thresh = 25  # 统计发现大多在0.22一下
        else:
            end3thresh = 0.3  # 原条件
            end2thresh = 20  # 原条件

        # 分闸门计算奖励
        # 注意这里要多加一个闸门　因为需要计算下游边界闸门的水位奖励
        for gate_id in range(self.n_gates + 1):
            reward = 0
            if stateflag3:
                # EndState = 3: 调整后出现部分值超出水动力模型范围，出现NAN情况
                for n in list(resultMessage.values()):
                    if np.any(np.isnan(n)):
                        raise Exception('水力学模型出现nan!!! \n\n{}'.format(resultMessage))
                        # endState = '3'
                        # reward = self.rewardTable['abs_20'] * self.n_gates
                        # return reward, endState

            if stateflag4:
                # 判断每个闸闸前水位2小时水位变幅是否超过15cm
                gate_num = str(self.start_gate + gate_id)
                if self.lastzqsw[gate_id] and resultMessage[gate_num + 'zqsw'] - self.lastzqsw[gate_id] > 0.15:
                    endState += '4' + gate_num + '-'
                    reward += self.rewardTable['abs_20']

            # if step>=4000:
            if stateflag5:
                # 判断每个闸闸前水位24小时水位变幅是否超过30cm
                gate_num = str(self.start_gate + gate_id)
                if len(self.zqswlist[gate_id]) >= 12:
                    zqswmax = max(self.zqswlist[gate_id])
                    zqswmin = min(self.zqswlist[gate_id])
                    if zqswmax - zqswmin > end3thresh:
                        endState += '5' + gate_num + '(%f)-' % (zqswmax - zqswmin)
                        reward += self.rewardTable['abs_20']
                    # TODO 明确这里是每24小时检查一次还是每个时间片检查前24小时的
                    self.zqswlist[gate_id].clear()

            if stateflag6:
                # 判断水位有没有超范围
                minzqsw, maxzqsw = utils.getzqswrange(self.start_gate, self.n_gates + 1)
                gate_num = str(self.start_gate + gate_id)
                if resultMessage[gate_num + 'zqsw'] < minzqsw[gate_id]:
                    value = abs(resultMessage[gate_num + 'zqsw'] - minzqsw[gate_id])
                    endState += '6' + gate_num + ' lower=(%f)-' % value
                    reward += self.rewardTable['abs_20']
                elif resultMessage[gate_num + 'zqsw'] > maxzqsw[gate_id]:
                    value = abs(resultMessage[gate_num + 'zqsw'] - maxzqsw[gate_id])
                    endState += '6' + gate_num + ' higher=(%f)-' % value
                    reward += self.rewardTable['abs_20']

            gate_num = str(self.start_gate + gate_id)
            abs_difference = abs(resultMessage[gate_num + 'zqsw'] - self.targetTable[gate_num]) * 100
            if 0 <= abs_difference < 5:
                reward += self.rewardTable['abs_0_5']
            elif 5 <= abs_difference < 10:
                reward += self.rewardTable['abs_5_10']
            elif 10 <= abs_difference < 15:
                reward += self.rewardTable['abs_10_15']
            elif 15 <= abs_difference < end2thresh:
                reward += self.rewardTable['abs_15_20']
            else:
                reward += self.rewardTable['abs_20']
                if config.END_STATE_2_CONSTRAINT:
                    endState += '2' + gate_num + '-'  # 调整后的水位差值在20以上，这是崩了的情况

            reward_list.append(reward)

        # print('6GATES:', reward_list)
        down_boundary_reward = reward_list[-1]
        reward_list = [reward + down_boundary_reward / (len(reward_list) - 1) for idx, reward in enumerate(reward_list)
                       if idx < self.n_gates]
        # reward_list[-1] += down_boundary_reward
        # 将下游边界闸门的奖励平均算在最后一个正常闸门上
        # print('5GATES:', reward_list, '\n')

        # 对调控次数进行赏罚
        # sumchange = sum(self.changeornot) / len(self.changeornot)
        # if sumchange < 0.4:
        #     reward += 0.01
        # elif sumchange < 0.8:
        #     reward -= 0.01
        # else:
        #     reward -= 0.03
        reward = sum(reward_list)

        if endState == '' and reward < self.rewardTable['abs_0_5'] * (self.n_gates + 1):
            endState = '1'

        if endState == '' and reward == self.rewardTable['abs_0_5'] * (self.n_gates + 1):
            endState = '0'

        # print('{:04d}: 各闸门reward: {} 总reward: {:.3f} {}'.format(self.env_id, reward_list, sum(reward_list), endState))

        return reward_list, endState

    def build_idle_action(self):
        """
        # 构造空调控指令 维持闸门位置不变
        :return:
        """
        idle_action = [self.mid for _ in range(self.n_gates)]
        return idle_action

    def get_init_state_from_INPUTFILE(self):
        """
        获取给定的初始状态
        :param self:
        :return:
        """
        message = OrderedDict()
        sjsw = utils.getsjsw(self.start_gate, self.n_gates + 1)
        # 读取INNERBOUNDARY中的初始闸口开度
        innerfile = os.path.join(self.INPUTFILE_PATH, 'INNERBOUNDARY.txt')
        f = open(innerfile, 'r', encoding='GBK')
        innerboundary = [lines for lines in f]
        init_zkkd = [int(ib) for ib in innerboundary[2].split()]
        initzqsw = []
        for i in range(self.n_gates):
            gtnum = str(self.start_gate + i)
            message[gtnum + 'zqsw'] = [sjsw[i]]
            initzqsw.append(message[gtnum + 'zqsw'][0])
            # self.acdict[int(len(self.acdict)/2)-1] # 现在在调控动作一半内 #self.acdict[len(self.acdict) - 1]
            actionrange = self.acdict[self.mid]
            message[gtnum + 'zkkd'] = np.array([
                [init_zkkd[i]], [init_zkkd[i]]
            ])

        # 下游边界闸门初始水位
        message[str(self.start_gate + self.n_gates) + 'zqsw'] = [sjsw[self.n_gates]]

        for i in range(1, len(initzqsw)):
            if (initzqsw[i] > initzqsw[i - 1]):
                print("back larger than front")
                break

        # print(initzqsw)
        return message

    def warm_up_model(self, total_warm_up_step, save_log, logger, log_root_dir, log_parse_interval):
        """
        预热模型 使水位 流量趋于稳定
        :param total_warm_up_step: 总共预热的轮次
        :param logger: 日志记录器
        :return:
        """
        # print('\n\n{} 模型预热 共{}轮'.format('=' * 20, total_warm_up_step))
        if save_log:
            systime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            sys_state_dir = os.path.join(log_root_dir, systime, 'state_log')
            output_dir = os.path.join(log_root_dir, systime, 'output_log')
            inputfile_log_dir = os.path.join(log_root_dir, systime, 'inputfile_log')

            os.makedirs(sys_state_dir)
            os.makedirs(output_dir)
            os.makedirs(inputfile_log_dir)

        # 转换STATE到0
        self.change_STATE_file(0)

        # self.reset(self.start_gate, self.n_gates)
        # initial data
        initState = self.get_init_state_from_INPUTFILE()  # 根据表格初始一套输入，只包含-7200, 0
        # self.acdict[self.mid]]是闸门不动，初始的环境信息, 为了初始化
        self.getNextState(initState, self.get_init_action(), 1e5)

        # utils.copy_waterdepth_to_result_time_0()
        # self.acdict[self.mid]]是闸门不动，初始的环境信息
        observation, _, _, _, newState, _ = self.getNextState(initState, self.get_init_action(), 1e5)
        # step=0
        for warmup_step in range(total_warm_up_step):
            # print('.', end='', flush=True)
            # 构造空action
            action = self.build_idle_action()
            # RL take action and get next observation and reward
            # 根据当前动作获得nextState，当前action得分，end状态
            next_observation, reward, endState, action, newState, log = self.getNextState(newState, action, warmup_step)
            if save_log:
                # 记录结果
                logger.set_log_data(
                    episode=warmup_step,
                    loss=0,
                    global_step=warmup_step,
                    ep_r=0,
                    reward=reward,
                    endState=endState,
                    log=log,
                    env=self,
                    action=action,
                    log_root_dir=log_root_dir,
                    systime=systime
                )
                logger.add_to_recorder()
                print(warmup_step, endState)

                # 保存INPUTFILE
                shutil.copytree(self.INPUTFILE_PATH,
                                os.path.join(inputfile_log_dir, '{:04d}_INPUTFILE'.format(warmup_step)))
                shutil.copytree(self.OUTPUTFILE_PATH,
                                os.path.join(inputfile_log_dir, '{:04d}_OUTPUTFILE'.format(warmup_step)))

        if save_log:
            logger.saveLog(output_dir, sys_state_dir)
            utils.analyse_log(os.path.join(output_dir, '{}.csv'.format(config.WARM_UP_STEP - 1)),
                              start_gate=config.START_GATE, num_gate=config.NUM_GATE)

        # 预热结束
        # 转换STATE到1
        self.change_STATE_file(1)
        return newState, None, None

    def copy_input_file_to_log(self, log_dir, target):
        """
        复制当前的inputfile到log
        :param target: 目标文件夹名称
        :return:
        """
        shutil.copytree(self.INPUTFILE_PATH, os.path.join(log_dir, target))

    def copy_waterdepth_to_result_time_0(self):  # change
        with open(os.path.join(self.OUTPUTFILE_PATH, 'Waterdepth.txt'), 'r', encoding='GBK') as f:
            data = f.readlines()
            f.close()
        data = utils.new_line_adapter(data)
        with open(os.path.join(self.INPUTFILE_PATH, 'RESULT_TIME_0.txt'), 'w', encoding='GBK')as f:
            f.writelines(data)
            f.close()

    def change_STATE_file(self, next_state):
        """
        将新的状态值写入STATE.txt
        :param next_state: 状态值
        :return:
        """
        with open(os.path.join(self.INPUTFILE_PATH, 'STATE.txt'), 'w') as file:
            # 适配换行符号
            lines = utils.new_line_adapter(['{}\n'.format(next_state)])
            file.writelines(lines)

    def _concat_observation_inlet(self, observation):
        """
        拼接观测量和分水表
        :param next_state: 观测量
        :return: 拼接之后的观测量
        """
        # 只有正式调控 作为特征时才拼接
        if self.cur_step < 0:
            return observation
        forward_length = config.INLET_FORWARD_LENGTH
        # 用多步的分水表拉平之后拼接成特征
        forward_inlet = self.inlet_table[self.cur_step:self.cur_step + forward_length, :]
        # 如果长度不足则用最后的补充长度
        pad = [forward_inlet[-1, :]] * (forward_length - len(forward_inlet))
        forward_inlet = np.concatenate([forward_inlet, pad], axis=0) if len(pad) else forward_inlet
        # 拼接观测量
        observation = np.hstack([observation, forward_inlet.flatten()])
        return observation

    def _build_inlet_varience(self):
        """
        生成分水表的变化
        :return:
        """
        assert self.cur_step < len(self.inlet_table), '运行步长大于分水表长度,需要检查'
        # 从分水表中读取inlet
        inlet = list(self.inlet_table[self.cur_step, :-1])

        # 如果设置分水扰动 则按照分水扰动构建分水表 否则按照分水表
        if config.INLET_DISTURBANCE:
            # 加扰动因子
            if self.cur_step % 12 == 0:
                disturbance_factor = config.DISTURBANCE_FACTOR
                for idx, factor in enumerate(disturbance_factor):
                    assert factor >= 0, '扰动因子必须为正数!'
                    # 计算当前分水口的最大扰动数值
                    disturbance_boundary = inlet[idx] * factor / 100
                    # 以分水表初始数值为基准 生成正态分布的噪声扰动
                    delta_noise = disturbance_boundary * np.random.normal(loc=0.0, scale=0.3, size=None)
                    # 限制噪声范围和小数点
                    delta_noise = max(-disturbance_boundary, min(delta_noise, disturbance_boundary))
                    # 加噪声
                    inlet[idx] = round(inlet[idx] + delta_noise, 3)

                assert len(inlet) == len(disturbance_factor)
            else:
                inlet = self.cur_inlet

        self.cur_inlet = inlet
        # 转换为输出文件中的行
        inlet = '\t'.join([str(v) for v in inlet]) + '\n'
        return inlet

    def _build_init_inlet(self):
        """
        返回初始第一个时间片的inlet
        :return: 初始第一个时间片的inlet
        """
        assert len(self.inlet_table), '分水表长度异常'
        self.cur_inlet = list(self.inlet_table[0, :-1])
        return '\t'.join([str(v) for v in self.inlet_table[0, :-1]]) + '\n'


if __name__ == '__main__':
    env = WaterModel(
        config.NUM_GATE,  # 闸门数
        config.START_GATE,  # 起始闸门
        env_id=9999,  # 环境进程id
    )
    for env.cur_step in range(1000):
        print(env.cur_step, env._build_inlet_varience())
