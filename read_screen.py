import cv2
import numpy as np
import pyautogui

# 设置捕获电脑区域
monitor = (600, 390, 400, 400)


# 创建 windows 窗体
# cv2.namedWindow("实时屏幕捕获".encode("gbk").decode('UTF-8', errors='ignore'), cv2.WINDOW_NORMAL)


def screen():
    screenshot = pyautogui.screenshot(region=monitor)
    screenshot_np = np.array(screenshot)
    # print(screenshot_np.shape)

    # 将 BGR 转换为 RGB (OpenCV 默认使用 RGB)
    screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)
    return screenshot_np

    # 显示屏幕截取画面
    # cv2.imshow("实时屏幕捕获".encode("gbk").decode('UTF-8', errors='ignore'), screenshot_np, )

    # 监控按键，按下 q 退出程序


def windows_test():
    cv2.namedWindow("实时屏幕捕获".encode("gbk").decode('UTF-8', errors='ignore'), cv2.WINDOW_NORMAL)

    while True:

        screenshot = pyautogui.screenshot(region=monitor)
        screenshot_np = np.array(screenshot)
        # print(screenshot_np.shape)

        # 将 BGR 转换为 RGB (OpenCV 默认使用 RGB)
        screenshot_np = cv2.cvtColor(screenshot_np, cv2.COLOR_BGR2GRAY)

        # 显示屏幕截取画面
        cv2.imshow("实时屏幕捕获".encode("gbk").decode('UTF-8', errors='ignore'), screenshot_np)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


# windows_test()
