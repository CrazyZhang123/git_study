import os
import glob

def replace_symbols_in_file(file_path):
    """替换单个文件中的符号"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 执行替换
        # 注意替换顺序，先替换\[和\]，再替换\(和\)，避免冲突
        content = content.replace(r'\[', '$').replace(r'\]', '$')
        content = content.replace(r'\(', '$').replace(r'\)', '$')
        # 替换\tag
        content = content.replace(r'\tag', '')
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"已处理: {file_path}")
        return True
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return False

def batch_replace_symbols(directory):
    """批量处理目录下的md和txt文件"""
    # 检查目录是否存在
    if not os.path.isdir(directory):
        print(f"错误: 目录 '{directory}' 不存在")
        return
    
    # 查找所有md和txt文件
    file_patterns = [
        os.path.join(directory, '*.md'),
        os.path.join(directory, '*.txt')
    ]
    
    files_to_process = []
    for pattern in file_patterns:
        files_to_process.extend(glob.glob(pattern))
    
    if not files_to_process:
        print(f"在目录 '{directory}' 中未找到md或txt文件")
        return
    
    # 处理每个文件
    print(f"找到 {len(files_to_process)} 个文件，开始处理...")
    success_count = 0
    for file_path in files_to_process:
        if replace_symbols_in_file(file_path):
            success_count += 1
    
    print(f"处理完成，成功处理 {success_count}/{len(files_to_process)} 个文件")

if __name__ == "__main__":
    # 可以直接指定目录，或者从命令行输入
    import sys
    
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
    else:
        # 如果未提供目录，使用当前工作目录
        target_directory = os.getcwd()
        print(f"未指定目录，将处理当前目录: {target_directory}")
    
    batch_replace_symbols(target_directory)
