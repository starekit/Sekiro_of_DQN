from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
import pyqtgraph as pg
from PyQt5.QtCore import QThread, pyqtSignal, QWaitCondition, QMutex, QMutexLocker
import numpy
import random
import time
import torch
import blood_read
from DQN_Net import DQN, ReplayBuffer
import keycontrol
from read_screen import screen
from torchvision import transforms
from pynput.keyboard import Controller
  

class AGENT(DQN):
    def __init__(self, gamma, replay_buffer):
        super().__init__(gamma, replay_buffer)

    def action_select(self, start_state):
        epsilon = 0.1

        if random.uniform(0, 1) < epsilon:
            action = random.randint(0, 3)
        else:
            action = torch.argmax(self.q.forward(start_state))

        return torch.tensor([action])  # 确保与torch.gather兼容


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.ui = uic.loadUi('ui/main.ui', self)
        except Exception as e:
            print(f"UI加载异常", {e})
            exit(-1)
        self.ui.start_button.clicked.connect(self.start)
        self.ui.stop_button.clicked.connect(self.stop)

        self.work = THREAD()
        self.X_data = []
        self.Y_data = []

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('w')
        #
        proxy_widget = QtWidgets.QGraphicsProxyWidget()
        proxy_widget.setWidget(self.plot_widget)
        #
        graphics_view = self.findChild(QtWidgets.QGraphicsView, 'loss')
        if graphics_view is None:
            print("未找到QGraphicsView")
        #
        scene = graphics_view.scene() if graphics_view.scene() else QtWidgets.QGraphicsScene()
        scene.addItem(proxy_widget)
        graphics_view.setScene(scene)

    @staticmethod
    def stop(self):
        exit(-1)

    def start(self):

        self.work.start()
        self.work.tigger.connect(self.display)

    def display(self, result1):
        if result1 != 0:
            self.X_data.append(len(self.X_data) + 1)
            self.Y_data.append(result1)
            # print("X:", self.X_data[len(self.X_data) - 1], "Y:", self.Y_data[len(self.Y_data) - 1])
            self.plot_widget.clearPlots()
            self.plot_widget.plot(self.X_data, self.Y_data, pen='b')
            self.plot_widget.setTitle('LOSS:')


class THREAD(QThread):
    tigger = pyqtSignal(float)

    def __init__(self, ):
        super(THREAD, self).__init__()
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self.paused = False
        self.agent1 = AGENT(gamma=0.01, replay_buffer=ReplayBuffer(2000))
        self.batch_size = 32
        self.reward = 0

    def run(self):
        sum_reward = 0
        number = 1
        picture = transforms.ToTensor()
        state = picture(screen()).cuda()

        while True:
            """获取boss及自身现在血量"""
            self_blood = blood_read.self_blood_count()
            boss_blood = blood_read.boss_blood_count()
            """依据目前状态计算并选择动作"""
            action = self.agent1.action_select(state)
            '''执行动作'''
            keycontrol.CONTROL().control(action)
            """获取boss及自身动作执行后血量"""
            now_self_blood = blood_read.self_blood_count()
            now_boss_blood = blood_read.boss_blood_count()
            """奖励判断"""
            if now_self_blood < self_blood:
                self.reward = -4
                sum_reward -= 4

            if now_boss_blood < boss_blood:
                self.reward = 2
                sum_reward += 2
            if now_self_blood == self_blood:
                self.reward = 1
                sum_reward += 1
            if now_boss_blood == boss_blood:
                self.reward = -1
                sum_reward -= 1
            if self_blood == 1:
                print("下一轮训练:")
                Next()

            """获取下一状态"""
            next_state = picture(screen()).cuda()
            """将目前状态，动作，下一状态，奖励组合成transition:tuple"""
            transition = (state, action, next_state, self.reward)
            # if len(replay_buffer) == 990:
            #     replay_buffer.popleft()
            """将transition加入经验池(replay_buffer)"""
            self.agent1.replay_buffer.push(transition=transition)
            """将下一状态赋值给上一状态(start_state)"""
            state = next_state

            """判断条件并更新网络"""
            if len(self.agent1.replay_buffer) > self.batch_size:
                transitions = self.agent1.replay_buffer.sample(self.batch_size)
                start_time = time.time()
                self.agent1.update(transitions)
                end_time = time.time()
                print("TrainTime:", end_time - start_time)

            if number % 5 == 0:
                self.agent1.q_target_update()

                # if now_self_blood == 0:
                break
            now_loss = numpy.array(self.agent1.loss)
            self.tigger.emit(now_loss)

    def resume(self):
        with QMutexLocker(self.mutex):
            self.paused = False
            self.resume_signal.emit()  # 发出信号唤醒线程
            self.wait_condition.wakeAll()


def Next():
    time.sleep(3)
    Controller().press('k')
    time.sleep(0.3)
    Controller().release('k')
    time.sleep(0.3)
    Controller().press('l')
    time.sleep(0.3)
    Controller().release('l')
    time.sleep(0.2)



if __name__ == "__main__":
    app = QApplication([])
    windows = MainWindow()
    windows.show()
    app.exec_()
