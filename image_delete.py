import os

class FileComparator:
    def __init__(self, image_folder, xml_folder):
        self.image_folder = image_folder
        self.xml_folder = xml_folder

    def compare_and_delete(self):
        image_files = set([os.path.splitext(file)[0] for file in os.listdir(self.image_folder) if file.endswith('.jpg')])
        xml_files = set([os.path.splitext(file)[0] for file in os.listdir(self.xml_folder) if file.endswith('.xml')])

        files_to_keep = image_files.intersection(xml_files)

        for file in os.listdir(self.image_folder):
            if file.endswith('.jpg') and os.path.splitext(file)[0] not in files_to_keep:
                os.remove(os.path.join(self.image_folder, file))

        for file in os.listdir(self.xml_folder):
            if file.endswith('.xml') and os.path.splitext(file)[0] not in files_to_keep:
                os.remove(os.path.join(self.xml_folder, file))

# 使用示例
image_folder = "/home/ymt/文档/xianlan/xianquan_img/"
xml_folder = "/home/ymt/文档/xianlan/xianquan_xml/"
# "D:/Data/Ymt/biaozhu/RF/30010ECR-c/"

comparator = FileComparator(image_folder, xml_folder)
comparator.compare_and_delete()
