import hg_net as net
import torch
import test_data_loader as loader
import numpy as np
import cv2
import shutil
import os
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
import dlib
import csv

# import fy_net as net
model = net.FAN(2)
# model_path = './model/model_29_6762.pkl'
model_path = './good_model/model_80_13525.pkl'
# model_path = './model/model_11_3387.pkl'
model.load_state_dict(torch.load(model_path))
save_path = './small_out_imgs/'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # cuda:0代表起始的； #device_id为0,如果直接是cuda,同样默认是从0开始，可以根据实际需要修改起始位置，如cuda:1

model.cuda(device)
sl1_criterion = nn.SmoothL1Loss().cuda()

path_photos_from_camera="data/data_faces_from_camera/"

existing_faces_cnt = 0  # 已录入的人脸计数器 / cnt for counting saved faces



# 读取带中文的图片，bgr
def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def cv_imwrite(file_path, img):
    cv2.imencode('.jpg', img)[1].tofile(file_path)


# 新建保存人脸图像文件和数据 CSV 文件夹 / Mkdir for saving photos and csv
def pre_work_mkdir():
    # 新建文件夹 / Create folders to save face images and csv
    if os.path.isdir(path_photos_from_camera):
        pass
    else:
        os.mkdir(path_photos_from_camera)


# 如果有之前录入的人脸, 在之前 person_x 的序号按照 person_x+1 开始录入 / Start from person_x+1
def check_existing_faces_cnt():
    global existing_faces_cnt
    if os.listdir("data/data_faces_from_camera/"):
        # 获取已录入的最后一个人脸序号 / Get the order of latest person
        person_list = os.listdir("data/data_faces_from_camera/")
        person_num_list = []
        for person in person_list:
            person_order = person.split('_')[1].split('_')[0]
            person_num_list.append(int(person_order))
        existing_faces_cnt = max(person_num_list)

    # 如果第一次存储或者没有之前录入的人脸, 按照 person_1 开始录入 / Start from person_1
    else:
        existing_faces_cnt = 0


def create_face_folder():
    global existing_faces_cnt,current_face_dir
    # 新建存储人脸的文件夹 / Create the folders for saving faces
    # 获取目录中的所有子目录名称
    subdirectories = next(os.walk(path_photos_from_camera))[1]

    # 检查子目录名称中是否存在包含指定字符串的目录
    exists = any(input_name_char in directory for directory in subdirectories)

    # 示例使用
    if exists:
        print("存在包含指定字符串的目录名称")
        current_face_dir = path_photos_from_camera + \
                                "person_" + str(existing_faces_cnt) + "_" + \
                                input_name_char
    else:
        existing_faces_cnt += 1
        if input_name_char:
            current_face_dir = path_photos_from_camera + \
                                    "person_" + str(existing_faces_cnt) + "_" + \
                                    input_name_char
        else:
            current_face_dir = path_photos_from_camera + \
                                    "person_" + str(existing_faces_cnt)
        os.makedirs(current_face_dir)


def val_test(image_path, user_name):
    global input_name_char
    input_name_char=user_name
    model.eval()
    ans = 0
    img = cv_imread(image_path)
    img_cpy_ori = img.copy()
    img_cpy = img.copy()
    # 将图像转换为灰度图像
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

    # 处理每个检测到的人脸
    face = faces[0]
    x, y, w, h = face[0], face[1], face[2], face[3]
    # 计算人脸区域的边长
    face_size = max(w, h)

    # 计算正方形人脸区域的左上角坐标
    x_square = x - (face_size - w) // 2
    y_square = y - (face_size - h) // 2

    # 计算正方形人脸区域的右下角坐标
    x_square_end = x_square + face_size
    y_square_end = y_square + face_size

    # 确保坐标不超出图像边界
    x_square = max(x_square, 0)
    y_square = max(y_square, 0)
    x_square_end = min(x_square_end, img_cpy.shape[1])
    y_square_end = min(y_square_end, img_cpy.shape[0])

    # 提取正方形人脸区域
    face_region = img_cpy[y_square:y_square_end, x_square:x_square_end]
    face_region = cv2.resize(face_region, (256, 256))

    # 进行相同的人脸处理代码
    img_cpy = face_region.copy()
    img = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # 转换为torch.float32类型
    img = img.astype(np.float32)

    # 使用torch.unsqueeze()在dim=0上添加两个额外的维度
    # 将img转换为PyTorch的Tensor
    img_tensor = torch.from_numpy(img)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    img_tensor /= 255.
    img = Variable(img_tensor).cuda()

    start_ = time.time()
    out, reg_outs = model(img)
    print(time.time() - start_)

    colors = [(255, 255, 0), (0, 255, 255)]
    k, reg = 0, reg_outs[0]
    reg = reg.cpu().data
    reg = np.array(reg)
    reg *= 4.0
    reg += 32.0
    reg *= 4.0
    reg = np.reshape(reg, (-1, 2))
    test_img = img_cpy.copy()

    if len(test_img.shape) == 2:
        test_img = test_img.copy()
    else:
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    for i in range(0, reg.shape[0]):
        cv2.rectangle(test_img, (int(reg[i, 0]) - 1, int(reg[i, 1]) - 1),
                      (int(reg[i, 0] + 1), int(reg[i, 1]) + 1), colors[k])

    pre_work_mkdir()
    check_existing_faces_cnt()
    create_face_folder()

    out_path = current_face_dir + '/' + str(ans) + '.jpg'
    cv_imwrite(out_path, img_cpy_ori)

    ans += 1
    out_path = current_face_dir +'/'+ str(ans) + '.jpg'
    cv_imwrite(out_path, test_img)

    save_to_csv(user_name,reg[:, 0].astype(str))


def save_to_csv(user_name, features_mean_personX):
    person_list = os.listdir("data/data_faces_from_camera/")
    person_list.sort()

    # 检查CSV文件是否存在
    if not os.path.isfile("data/features_all.csv"):
        # 如果文件不存在，创建新文件并写入表头
        with open("data/features_all.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["empty"] + ["Feature"]*68)  # 表头包括128个特征列
            print("Created new CSV file")

    with open("data/features_all.csv", "r") as file:
        rows = list(csv.reader(file))

    for person in person_list:
        if user_name in person:
            found_user = False  # 标记是否找到匹配的用户

            with open("data/features_all.csv", "w", newline="") as csvfile:
                writer = csv.writer(csvfile)

                for row in rows:
                    if row[0] == user_name:
                        # 更新特定用户名的行的features_mean_personX
                        if len(row) == 1:
                            # "person_x"
                            person_name = row[0]
                        else:
                            # "person_x_tom"
                            person_name = row[0].split('_', 2)[-1]
                        features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
                        writer.writerow(features_mean_personX)
                        found_user = True
                    else:
                        # 保持其他行不变
                        writer.writerow(row)

                if not found_user:
                    # 如果没有找到匹配的用户，添加新行
                    person_name = user_name
                    features_mean_personX = np.insert(features_mean_personX, 0, person_name, axis=0)
                    writer.writerow(features_mean_personX)


# 加载人脸检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')


def val_test_camera():
    model.eval()
    ans = 0

    # 打开摄像头
    cap = cv2.VideoCapture(0)  # 根据摄像头索引调整参数，例如使用0表示第一个摄像头

    while True:
        ret, frame = cap.read()  # 读取视频帧

        # 将图片转换为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # 处理每个检测到的人脸
        for face in faces:
            x, y, w, h = face[0], face[1], face[2], face[3]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 计算人脸区域的边长
            face_size = max(w, h)

            # 计算正方形人脸区域的左上角坐标
            x_square = x - (face_size - w) // 2
            y_square = y - (face_size - h) // 2

            # 计算正方形人脸区域的右下角坐标
            x_square_end = x_square + face_size
            y_square_end = y_square + face_size

            # 确保坐标不超出图像边界
            x_square = max(x_square, 0)
            y_square = max(y_square, 0)
            x_square_end = min(x_square_end, frame.shape[1])
            y_square_end = min(y_square_end, frame.shape[0])

            # 提取正方形人脸区域
            face_region = frame[y_square:y_square_end, x_square:x_square_end]
            face_region = cv2.resize(face_region, (256, 256))

            # 进行相同的人脸处理代码
            img_cpy = face_region.copy()
            img = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            img_tensor = torch.from_numpy(img)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
            img_tensor /= 255.
            img = Variable(img_tensor).cuda()

            start_ = time.time()
            out, reg_outs = model(img)
            print(time.time() - start_)

            colors = [(255, 255, 0), (0, 255, 255)]
            k, reg = 0, reg_outs[0]
            reg = reg.cpu().data
            reg = np.array(reg)
            reg *= 4.0
            reg += 32.0
            reg *= 4.0
            reg = np.reshape(reg, (-1, 2))
            test_img = img_cpy.copy()

            if len(test_img.shape) == 2:
                test_img = test_img.copy()
            else:
                test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

            for i in range(0, reg.shape[0]):
                cv2.rectangle(test_img, (int(reg[i, 0]) - 1, int(reg[i, 1]) - 1),
                              (int(reg[i, 0] + 1), int(reg[i, 1]) + 1), colors[k])
            out_imgs = save_path + str(ans) + '.jpg'
            cut = 0
            ans += 1
            cv_imwrite(out_imgs, test_img)

        # 显示视频帧
        cv2.imshow('Video', frame)

        # 按下'q'键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头和关闭窗口
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_list = './256_Data_test/data_tag.list'
    test_data = loader.getDataFromList(test_list)
    test_dataset = loader.DataLoader(test_data, 256, 68)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    image_path='face_dataset/Natalie Portman/087_dabfb9e0.jpg'
    user_name = 'Natalie'

    val_test(image_path, user_name)
