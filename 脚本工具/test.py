import cv2

# ===================== 实时摄像头测试 =====================
try:
    # 打开摄像头（0 表示默认摄像头，笔记本内置摄像头一般是 0）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("无法打开摄像头，请检查摄像头是否被占用")
    
    print("✅ 摄像头已打开，按 Q 键退出")
    
    while True:
        # 读取一帧画面
        ret, frame = cap.read()
        if not ret:
            break
        
        # 实时处理：灰度化 + 边缘检测
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_canny = cv2.Canny(frame_gray, 50, 150)
        
        # 显示窗口
        cv2.imshow("实时摄像头画面", frame)
        cv2.imshow("实时边缘检测", frame_canny)
        
        # 按 Q 键退出（注意：窗口要处于激活状态）
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 摄像头测试完成，已正常退出")

except Exception as e:
    print(f"❌ 摄像头测试失败：{e}")
    print("👉 排查建议：")
    print("   1. 检查摄像头是否被其他软件占用（如微信/钉钉）")
    print("   2. 管理员权限运行 VS Code")
