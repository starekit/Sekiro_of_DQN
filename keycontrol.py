import time
from pynput.keyboard import Controller, Key

'control class'


class ACTIONS(Controller):

    def __init__(self):
        super().__init__()

    def attack(self):
        self.press('k')
        time.sleep(0.2)
        self.release('k')

    def forward(self):
        self.press('w')
        time.sleep(0.2)
        self.release('w')
        # time.sleep(0.2)

    def backward(self):
        self.press('s')
        time.sleep(0.2)
        self.release('s')
        # time.sleep(0.2)

    def l_shift(self):
        self.press(Key.shift_l)
        time.sleep(0.2)
        self.release(Key.shift_l)
        # time.sleep(0.2)

    def r_shift(self):
        self.press(Key.shift_r)
        time.sleep(0.2)
        self.release(Key.shift_r)
        # time.sleep(0.2)

    def R_key(self):
        self.press('r')
        time.sleep(0.2)
        self.release('r')
        # time.sleep(0.2)

    def SPACE(self):
        self.press(Key.space)
        time.sleep(0.2)
        self.release(Key.space)
        # time.sleep(0.2)

    def M_key(self):
        self.press('m')
        time.sleep(0.2)
        self.release('m')
        # time.sleep(0.2)

    def J_key(self):
        self.press('j')
        time.sleep(0.2)
        self.release('j')
        # time.sleep(0.2)

    def Q_key(self):
        self.press('q')
        time.sleep(0.2)
        self.release('q')
        # time.sleep(0.2)


class CONTROL(ACTIONS):
    def __init__(self):
        super().__init__()

    def control(self, number):
        match number:
            # if number == 0:
            #     self.attack()
            # if number == 1:
            #     self.SPACE()
            # if number == 2:
            #     self.l_shift()
            # if number == 3:
            #     self.Q_key()
            # if number == 4:
            #     self.J_key()
            case 0:
                self.attack()
            case 1:
                self.SPACE()
            case 2:
                self.l_shift()
            case 3:
                self.Q_key()
            case 4:
                self.J_key()
            case 5:
                self.R_key()
            case 6:
                self.forward()
            case 7:
                self.backward()
            case _:
                print("ERROR")
        return number


if __name__ == '__main__':
    new = CONTROL()
