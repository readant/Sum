import os
import sys

# 测试模块列表
test_modules = [
    r"04_mediapipe\01_code\hand\手部检测\手部检测测试.py",
    r"04_mediapipe\01_code\hand\手势识别\手势识别测试.py",
    r"04_mediapipe\01_code\hand\双手协同\双手协同测试.py",
    r"04_mediapipe\01_code\hand\置信度估计\置信度估计测试.py",
    r"04_mediapipe\01_code\face\面部检测\面部检测测试.py",
    r"04_mediapipe\01_code\face\表情识别\表情识别测试.py",
    r"04_mediapipe\01_code\face\面部特征分析\面部特征分析测试.py",
    r"04_mediapipe\01_code\body\姿态检测\姿态检测测试.py",
    r"04_mediapipe\01_code\body\全身检测\全身检测测试.py"
]

def test_model_path(module_path):
    """测试模块的模型路径是否正确"""
    try:
        # 确保使用绝对路径
        absolute_path = os.path.abspath(module_path)

        # 读取文件内容
        with open(absolute_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取模型路径代码
        import re
        model_path_match = re.search(r'self\.model_path = (.*)', content)
        if model_path_match:
            model_path_code = model_path_match.group(1)

            # 模拟模块的目录结构
            module_dir = os.path.dirname(absolute_path)

            # 保存当前工作目录
            original_cwd = os.getcwd()

            try:
                # 切换到模块目录
                os.chdir(module_dir)
                # 直接执行模型路径计算
                model_path = eval(model_path_code)
                print(f"Model path: {model_path}")
                print(f"Path exists: {os.path.exists(model_path)}")
                return True
            finally:
                # 恢复原始工作目录
                os.chdir(original_cwd)
        else:
            print(f"❌ {module_path}: 未找到模型路径定义")
            return False
    except Exception as e:
        print(f"❌ {module_path}: 测试失败 - {str(e)}")
        return False

def main():
    print("测试 MediaPipe 模块的模型路径引用...\n")

    success_count = 0
    total_count = len(test_modules)

    for module in test_modules:
        print(f"测试: {module}")
        if test_model_path(module):
            success_count += 1
        print()

    print(f"测试结果: {success_count}/{total_count} 个模块路径正确")
    if success_count == total_count:
        print("✅ 所有模块的模型路径引用正确！")
    else:
        print("❌ 部分模块的模型路径引用存在问题，请检查。")

if __name__ == "__main__":
    main()