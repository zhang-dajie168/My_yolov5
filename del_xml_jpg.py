import os


def delete_jpg_without_xml(jpg_folder, xml_folder):
    # 获取jpg文件夹中的所有文件
    jpg_files = os.listdir(jpg_folder)

    # 获取xml文件夹中的所有文件的文件名（不包括扩展名）
    xml_files = [os.path.splitext(file)[0] for file in os.listdir(xml_folder)]

    # 遍历jpg文件夹中的文件
    for jpg_file in jpg_files:
        # 提取jpg文件的文件名（不包括扩展名）
        jpg_filename = os.path.splitext(jpg_file)[0]

        # 如果jpg文件的文件名不在xml文件名列表中，则删除该jpg文件
        if jpg_filename not in xml_files:
            jpg_path = os.path.join(jpg_folder, jpg_file)
            os.remove(jpg_path)
            print(f"Deleted: {jpg_path}")




# 指定jpg文件夹和xml文件夹的路径
jpg_folder = "/home/ymt/data-ZXJC/data-HW/0411-厚威/30010DXT-001/30010DXT-001-1_img"
xml_folder = "/home/ymt/data-ZXJC/data-HW/0411-厚威/30010DXT-001/Annotations"

# 删除jpg文件夹中没有对应xml文件的jpg文件
delete_jpg_without_xml(jpg_folder, xml_folder)
