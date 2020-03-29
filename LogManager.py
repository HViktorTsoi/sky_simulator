import pdb

import pandas as pd
import os

import utils


class LogManager(object):
    def __init__(self):
        self.sys_state_recorder = []
        self.output_recorder = []

        self.episode = None
        self.loss = None
        self.global_step = None
        self.ep_r = None
        self.reward = None
        self.endState = None
        self.log = None
        self.env = None
        self.action = None
        self.log_root_dir = None
        self.systime = None

    def set_log_data(self, episode, loss, global_step, ep_r, reward, endState, log, env, action, log_root_dir, systime):
        self.episode = episode
        self.loss = loss
        self.global_step = global_step
        self.ep_r = ep_r
        self.reward = reward
        self.endState = endState
        self.log = log
        self.env = env
        self.action = action
        self.log_root_dir = log_root_dir
        self.systime = systime

    def addREADME(self):
        txtPath = os.path.join(self.log_root_dir, self.systime,
                               'ngates_{}--start_{}.txt'.format(self.env.n_gates, self.env.start_gate))
        if os.path.exists(txtPath) is False:
            with open(txtPath, 'w') as f:
                f.close()

    def log_console_printer(self):
        print("------- episode {} --> global_step {} --> loss {:.8f}   reward: {}/{}  endState: {}------".format(
            self.episode, self.global_step, self.loss, self.ep_r, self.reward, self.endState))
        maxzkkd = 0
        minzkkd = 9999
        lastzkkd = 0
        zkkd_diff = []
        for i in range(self.env.n_gates):
            gtnum = str(self.env.start_gate + i)
            currentzkkd = self.log[str(gtnum) + 'zkkd'][0]
            if maxzkkd < currentzkkd:
                maxzkkd = currentzkkd
            if minzkkd > currentzkkd:
                minzkkd = currentzkkd
            # 断言闸口开度diff小于500
            # assert i == 0 or (i > 0 and abs(currentzkkd - lastzkkd) <= 500), 'zkkd > 500!!!'
            if i > 0 and abs(currentzkkd - lastzkkd) > 500:
                print('zkkd > 500!!!')
                pdb.set_trace()
            zkkd_diff.append(currentzkkd - lastzkkd)
            lastzkkd = currentzkkd
            print(
                '{0}zqsw: {1:.3f} / {0}zkkd: {2} '.format(gtnum, self.log[str(gtnum) + 'zqsw'],
                                                          self.log[str(gtnum) + 'zkkd'][0]))

        print("maxzkkd-minzkkd=", maxzkkd - minzkkd, " zkkd_diff=", zkkd_diff)

    def add_to_recorder(self):
        log = self.log
        sjsw = utils.getsjsw(self.env.start_gate, self.env.n_gates + 1)

        self.sys_state_recorder.append(
            [self.global_step, self.loss, self.action, self.ep_r, self.reward, self.endState])

        tmp = [self.global_step]
        # 处理多余的边界闸门log
        log[str(self.env.start_gate + self.env.n_gates) + 'zhsw'] = \
            log[str(self.env.start_gate + self.env.n_gates) + 'zqsw']
        log[str(self.env.start_gate + self.env.n_gates) + 'll'] = log['down_ll']
        log[str(self.env.start_gate + self.env.n_gates) + 'zkkd'] = [600]
        # 所有闸门的
        for i in range(self.env.n_gates + 1):
            gtnum = str(self.env.start_gate + i)
            tmp.extend([log[gtnum + 'zqsw'], log[gtnum + 'zhsw'], log[gtnum + 'zkkd'][0], log[gtnum + 'll'], sjsw[i],
                        log[gtnum + 'zqsw'] - sjsw[i]])
        tmp.extend([
            self.action, self.reward,
            log['up_zqsw'], log['down_ll'], log['inlet']
        ])

        self.output_recorder.append(tmp)

    def saveLog(self, output_dir, sys_state_dir):
        self.addREADME()

        log = self.log
        sjsw = utils.getsjsw(self.env.start_gate, self.env.n_gates)

        colunms = ['global_step']
        for i in range(self.env.n_gates + 1):
            gtnum = str(self.env.start_gate + i)
            colunms.extend(
                [gtnum + 'zqsw', gtnum + 'zhsw', gtnum + 'zkkd', gtnum + 'll', gtnum + 'sjsw', gtnum + 'difference'])
        colunms.extend([
            'actions', 'reward',
            'bound_up_zqsw', 'bound_down_ll',
            'inlet'
        ])

        pd.DataFrame(self.output_recorder,
                     columns=colunms).to_csv(
            os.path.join(output_dir, str(self.episode) + '.csv'), mode='a', index=False)
        self.output_recorder.clear()

        pd.DataFrame(self.sys_state_recorder,
                     columns=['global_step', 'loss', 'action', 'total_reward', 'delta_reward',
                              'end_state']).to_csv(
            os.path.join(sys_state_dir, '{}_{}.csv'.format(str(self.episode), self.endState)), mode='a', index=False)
        str(self.episode) + '.csv'
        self.sys_state_recorder.clear()
