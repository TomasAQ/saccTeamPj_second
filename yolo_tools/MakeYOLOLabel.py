import numpy as np
import pandas as pd
import data
import matplotlib.pyplot as plt
import cv2
import os

# 박스 만드는 코드
# def makeBox(start_no ,end_no ,  image):
#     image = cv2.imread(image, cv2.IMREAD_COLOR)
#     height , width , col = image.shape
#
#     for  no in range(start_no , end_no) :
#         bbox_left = df.loc[[no], ['bbox_left']]
#         bbox_top = df.loc[[no], ['bbox_top']]
#         bbox_right = df.loc[[no], ['bbox_right']]
#         bbox_bottom = df.loc[[no], ['bbox_bottom']]
#
#         valx = int(round(np.median(np.array([float(bbox_left['bbox_left']), float(bbox_right['bbox_right'])])), 0))
#         valy = int(round(np.median(np.array([float(bbox_top['bbox_top']), float(bbox_bottom['bbox_bottom'])])), 0))
#
#         label_x = valx / width
#         label_y = valy / height
#         label_width = int( round( abs(float(bbox_right['bbox_right']) - float(bbox_left['bbox_left'])), 0) ) / width
#         label_height = int( round( abs(float(bbox_bottom['bbox_bottom']) - float(bbox_top['bbox_top'])), 0) ) / height
#
#         print(label_x , label_y , label_width , label_height)
#
#         cv2.circle( image , (valx,valy ),1 , (0,255,255) , -1)
#         cv2.rectangle( image, ( bbox_left['bbox_left'] , bbox_top['bbox_top'] ) , ( bbox_right['bbox_right'], bbox_bottom['bbox_bottom']), (0, 255, 0), 3 )
#
#
#     cv2.imwrite('./data/saveimag.png', image)
#     cv2.imshow("image", image)
#     cv2.waitKey()
#     cv2.destroyAllWindows()

# 파일을 만들어서 저장
def saveLabeltxt(file_num , image):
    image_location = image
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    height, width, col = image.shape
    df_lab = df.loc[df['file_num'] == file_num]
    file_name = str(file_num).zfill(6) + '.txt'

    # df_lab_iloc = df.iloc[:,16] == file_num
    # print(df_lab_iloc)
    # tFchek = df_lab['type'].isin(['Car', 'Van', 'Truck'])
    # print(tFchek.count())
    # print(tFchek)
    #
    #
    # #print(tFchek.value_counts(normalize = True)[0] )
    #
    # check = tFchek.value_counts(normalize=True)[0]
    #
    # if check == 1.0 :
    #     if tFchek[0] == False :
    #         print(111111)
    #     elif tFchek[0] == True :
    #         print(2222)
    # #
    # # if tFchek.value_counts()[0] == tFchek.count():
    # #     return
    # # else :
    # #     print(2222222)
    #

    with open('./dataset/labels/' + file_name, 'w') as f:
    
        #label_x , label_y , label_width , label_height , label_type , file_num = makelabel(df_lab , height , width)

        # Yolo_label 데이터로 변경
        for i in range(df_lab.count()[0]):
            df_lab_i1 = df_lab.iloc[i]

            label_type = df_lab_i1[1]
            bbox_left = df_lab_i1[5]
            bbox_top = df_lab_i1[6]
            bbox_right = df_lab_i1[7]
            bbox_bottom = df_lab_i1[8]
            file_num = df_lab_i1[16]
            #print(type , bbox_left , bbox_top , bbox_right , bbox_bottom , file_num)

            valx = int(round(np.median(np.array([float(bbox_left), float(bbox_right)])), 0))
            valy = int(round(np.median(np.array([float(bbox_top), float(bbox_bottom)])), 0))

            label_x = round(valx / width, 6)
            label_y = round(valy / height, 6)
            #print(label_x , label_y)

            label_width = round(int(round(abs(float(bbox_right) - float(bbox_left)), 0)) / width, 6)
            label_height = round(int(round(abs(float(bbox_bottom) - float(bbox_top)), 0)) / height, 6)

            #print(label_x , label_y , label_width , label_height)
            #return label_x , label_y , label_width , label_height , label_type , file_num

            # 파일 저장
            if label_type == 'Car':
                f.write('0' + ' ' + '{:<06f}'.format(label_x) + " " + '{:<06f}'.format(label_y) + " " + '{:<06f}'.format(label_width) + " " + '{:<06f}'.format(label_height) + '\n')
            elif label_type == 'Van':
                f.write('1' + ' ' + '{:<06f}'.format(label_x) + " " + '{:<06f}'.format(label_y) + " " + '{:<06f}'.format(label_width) + " " + '{:<06f}'.format(label_height) + '\n')
            elif label_type == 'Truck':
                f.write('2' + ' ' + '{:<06f}'.format(label_x) + " " + '{:<06f}'.format(label_y) + " " + '{:<06f}'.format(label_width) + " " + '{:<06f}'.format(label_height) + '\n')
    
    # 빈 파일 삭제
    null_fileRemove(file_name , image_location)


def null_fileRemove(file_name , image_location):
    file = './dataset/labels/' + file_name

    img = image_location
    
    if os.path.isfile(file):
        f = open( file , 'r')
        data = f.readline()
        f.close()
        if data =='':
            # label 지우기
            os.remove(file)
            # 이미지 제거
            os.remove(img)


#
#
#
# def makelabel(df_lab , height , width):
#     print(df_lab.count()[0])
#     for i in range(df_lab.count()[0]):
#         df_lab_i1 = df_lab.iloc[i]
#
#         label_type = df_lab_i1[1]
#         bbox_left = df_lab_i1[5]
#         bbox_top = df_lab_i1[6]
#         bbox_right = df_lab_i1[7]
#         bbox_bottom = df_lab_i1[8]
#         file_num = df_lab_i1[16]
#         #print(type , bbox_left , bbox_top , bbox_right , bbox_bottom , file_num)
#
#         valx = int(round(np.median(np.array([float(bbox_left), float(bbox_right)])), 0))
#         valy = int(round(np.median(np.array([float(bbox_top), float(bbox_bottom)])), 0))
#
#         label_x = round(valx / width, 6)
#         label_y = round(valy / height, 6)
#         #print(label_x , label_y)
#
#         label_width = round(int(round(abs(float(bbox_right) - float(bbox_left)), 0)) / width, 6)
#         label_height = round(int(round(abs(float(bbox_bottom) - float(bbox_top)), 0)) / height, 6)
#
#         #print(label_x , label_y , label_width , label_height)
#         #return label_x , label_y , label_width , label_height , label_type , file_num
#
#
        


if __name__ == '__main__':
    # 출력 세팅
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_row', 500)
    pd.set_option('display.max_columns', 100)
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)
    df = pd.read_csv("data/tot_label.csv")
    #print(df)

    #
    #makeBox(51683,51689,'./data/007453.png')

    #
    for i in range(7481) :
        saveLabeltxt(i,'./dataset/images/'+str(i).zfill(6)+'.png')


# 구해야 하는 값 (class  ,  x , y , width , height)
    #                   1    , 0.326172 , 0.509371  , 0.036719  , 0.062918

# 1. 이미지 크기(가로,세로)를 가져온다.
# 2. class에 중심점을 찾는다.
# 3. 비율 계산하기

'''
       Unnamed: 0        type  truncated  occluded  alpha  bbox_left  bbox_top  bbox_right  bbox_bottom  dimensions_height  dimensions_width  dimensions_length  location_x  location_y  location_z  rotation_y  file_num

'''
