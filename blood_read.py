import pyautogui
import cv2

import numpy as np

monitor1 = (90, 852, 273, 10)
monitor2 = (94, 118, 272, 10)
# monitor2 = (367, 215, 273, 8)

'''获取自身血量'''


def self_blood_count():
    screen_self = pyautogui.screenshot(region=monitor1)
    screen_self_np = np.array(screen_self)
    screen_self_np = cv2.cvtColor(screen_self_np, cv2.COLOR_BGR2GRAY)

    self_blood = 0
    now = screen_self_np[5, :]
    for self_blood_num in range(120):
        if 60 < now[self_blood_num] < 98:
            self_blood += 1
    return self_blood


'''获取boss血量'''


def boss_blood_count():  #

    screen_boss = pyautogui.screenshot(region=monitor2)
    screen_np_boss = np.array(screen_boss)
    screen_np_boss = cv2.cvtColor(screen_np_boss, cv2.COLOR_BGR2GRAY)

    boss_blood = 0
    now_boss = screen_np_boss[6, :]
    # print(now_boss)
    for boss_blood_num in range(272):
        if now_boss[boss_blood_num] < 100:
            boss_blood += 1
        else:
            break
    return boss_blood


'''窗口测试'''


def windows_test():
    while True:
        screen_host = pyautogui.screenshot(region=monitor1)
        screen_host2 = pyautogui.screenshot(region=monitor2)

        screen_np = np.array(screen_host)
        screen_np2 = np.array(screen_host2)

        screen_np = cv2.cvtColor(screen_np, cv2.COLOR_BGR2GRAY)
        screen_np2 = cv2.cvtColor(screen_np2, cv2.COLOR_BGR2GRAY)

        cv2.imshow("TXT", screen_np)
        cv2.imshow("TXT2", screen_np2)

        boss_blood = boss_blood_count()
        self_blood = self_blood_count()
        print(boss_blood, self_blood)
        if cv2.waitKey(1) & 0xFF == 'q':
            break
    cv2.destroyAllWindows()

# windows_test()
