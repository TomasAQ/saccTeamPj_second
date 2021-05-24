import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

'''
# 컬럼

아래 참조 + 컬럼 하나 추가
 
 

#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters) 
   
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   
   
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
    
   파일 확인을 위한 파일 번호 추가
    file_num 
    
'''


# label 데이터 읽어 오기
def readData(fileName) :
    # 출력 세팅
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_row', 500)
    pd.set_option('display.max_columns', 100)
    # 경로 및 파일명 지정
    fileName = str(fileName).zfill(6)
    path = open("data/path.txt", 'r', encoding='utf-8').readline()

    data = pd.read_csv(path + "\\" + fileName + ".txt", sep=' ' , header=None )
    # 파일을 확인을 위한 파일명 추가
    data["file_num"] = fileName
    data.columns =test_columns
    return data

# 전체 label를 하나에 csv 파일에 저장
# 1~7481
def saveCSV(num) :
    tot_df = pd.DataFrame(columns=test_columns)

    for no in range(num):
        tot_df = tot_df.append([readData(no)])
    # 파일 내보내기
    tot_df.to_csv("data/tot_label.csv", mode='w')

    
def car_analy(df):
    # occluded 를 통한 비교

    # 80 % : 22993.600000000002 = 22994
    # 20% : 5748.400000000001 = 5748
    print("----------------------------------------------------------------------------------")
    print("전체 car : " , df.loc[(df['type'] == 'Car'), ['type']].count()[0])  # 28742
    print("occluded==0 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 0), ['type']].count()[0])  # 13457
    print("occluded==1 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 1), ['type']].count()[0])  # 8184
    print("occluded==2 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 2), ['type']].count()[0])  # 6173
    print("occluded==3 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 3), ['type']].count()[0])  # 928
    print("----------------------------------------------------------------------------------")

    print("전체 car 파일 0 ~ 5971 사이 존재 개수 : ", df.loc[ (df['type'] == 'Car') & (df['file_num'] <=5971 ) , ['type']].count()[0])
    print("전체 car 파일 5971 ~  사이 존재 개수 : ",df.loc[(df['type'] == 'Car') & (df['file_num'] > 5971), ['type']].count()[0])
    # print(df.loc[(df['type'] == 'Car') & (df['file_num'] <=5971), ['type','file_num','occluded'] ])


def type_ScatterPlot01():

    df_car = df.loc[(df['type'] == 'Car')]
    df_van = df.loc[(df['type'] == 'Van')]
    df_truck = df.loc[(df['type'] == 'Truck')]

    plt.scatter( 'location_x' ,'location_y' , data =df_car ,  marker='1'  , label='Car')
    plt.scatter('location_x', 'location_y', data=df_van,  marker='2' , label='Van')
    plt.scatter('location_x', 'location_y', data=df_truck,  marker='3' , label='Truck')
    plt.xlabel('location_x')
    plt.ylabel('location_y')

    plt.legend()
    plt.show()


def type_ScatterPlot02():


    df_car = df.loc[(df['type'] == 'Car') & (df['occluded'] == 0)]
    df_van = df.loc[(df['type'] == 'Van') & (df['occluded'] == 0)]
    df_truck = df.loc[(df['type'] == 'Truck') & (df['occluded'] == 0)]

    plt.scatter( 'location_x' ,'location_y' , data =df_car ,  marker='1'  , label='Car')
    plt.scatter('location_x', 'location_y', data=df_van,  marker='2' , label='Van')
    plt.scatter('location_x', 'location_y', data=df_truck,  marker='3' , label='Truck')
    plt.xlabel('location_x')
    plt.ylabel('location_y')

    plt.legend()
    plt.show()

def type_ScatterPlot03():

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    df_car = df.loc[(df['type'] == 'Car') & (df['occluded'] == 3)]
    df_van = df.loc[(df['type'] == 'Van') & (df['occluded'] == 3)]
    df_truck = df.loc[(df['type'] == 'Truck') & (df['occluded'] == 3)]

    ax.scatter('location_x', 'location_y','location_z' , data =df_car ,  marker='1'  , label='Car')
    ax.scatter('location_x', 'location_y','location_z' ,  data=df_van,  marker='2' , label='Van')
    ax.scatter('location_x', 'location_y','location_z' ,  data=df_truck,  marker='3' , label='Truck')
    ax.xlabel('location_x')
    ax.ylabel('location_y')
    ax.zlabel('location_z')

    plt.legend()
    plt.show()

def type_ScatterPlot_3D_01():
    df_car = df.loc[(df['type'] == 'Car') ]
    df_van = df.loc[(df['type'] == 'Van') ]
    df_truck = df.loc[(df['type'] == 'Truck') ]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(0, 0, 0, c='black', s=100)
    ax.scatter(df_car['location_x'], df_car['location_y'], df_car['location_z'])
    ax.scatter(df_van['location_x'], df_van['location_y'], df_van['location_z'])
    ax.scatter(df_truck['location_x'], df_truck['location_y'], df_truck['location_z'])

    ax.set_xlabel('location_x')
    ax.set_ylabel('location_y')
    ax.set_zlabel('location_z')
    plt.show()


def type_ScatterPlot_3D_02():
    df_car = df.loc[(df['type'] == 'Car') & (df['occluded'] == 3)]
    df_van = df.loc[(df['type'] == 'Van') & (df['occluded'] == 3)]
    df_truck = df.loc[(df['type'] == 'Truck') & (df['occluded'] == 3)]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(0, 0, 0 , c='black' ,  s=100)
    ax.scatter(df_car['location_x'],df_car['location_y'],df_car['location_z'])
    ax.scatter(df_van['location_x'], df_van['location_y'], df_van['location_z'])
    ax.scatter(df_truck['location_x'], df_truck['location_y'], df_truck['location_z'])

    ax.set_xlabel('location_x')
    ax.set_ylabel('location_y')
    ax.set_zlabel('location_z')
    plt.show()


def Scatter_plot_on_polar_axis():
    #df_car1 = df.loc[(df['type'] == 'Car') & (df['occluded'] == 0)]
    #df_car2 = df.loc[(df['type'] == 'Car') & (df['occluded'] == 3)]
    df_car = df.loc[(df['type'] == 'Car') ]
    df_van = df.loc[(df['type'] == 'Van') ]
    df_truck = df.loc[(df['type'] == 'Truck') ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(df_car['location_x'], df_car['location_y'] , alpha=0.75 , c='red')
    ax.scatter(df_van['location_x'], df_van['location_y'], alpha=0.75 , c='green')
    ax.scatter(df_truck['location_x'], df_truck['location_y'], alpha=0.75, c='blue')

    plt.show()

def type_label():
    print(df['type'].value_counts())

    #print(type(df['type'][0]))

    print(df['type'][1])

    #print(df['type'].value_counts()[1])

    names = [ 'Car' , 'DontCare' , 'Pedestrian' , 'Van' , 'Cyclist' , 'Truck' , 'Misc' , 'Tram' , 'Person_sitting']

    height = [df['type'].value_counts()[0], df['type'].value_counts()[1], df['type'].value_counts()[2], df['type'].value_counts()[3], df['type'].value_counts()[4] , df['type'].value_counts()[5] , df['type'].value_counts()[6], df['type'].value_counts()[7], df['type'].value_counts()[8]]
    y_pos = np.arange(len(names))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, names )
    #plt.xticks(y_pos, names, fontweight='bold')
    plt.show()



def occludedChk(df):
    # occluded 를 통한 비교

    # 80 % : 22993.600000000002 = 22994
    # 20% : 5748.400000000001 = 5748
    # print("----------------------------------------------------------------------------------")
    # print("전체 car : " , df.loc[(df['type'] == 'Car'), ['type']].count()[0])  # 28742
    # print("occluded==0 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 0), ['type']].count()[0])  # 13457
    # print("occluded==1 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 1), ['type']].count()[0])  # 8184
    # print("occluded==2 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 2), ['type']].count()[0])  # 6173
    # print("occluded==3 인 car", df.loc[(df['type'] == 'Car') & (df['occluded'] == 3), ['type']].count()[0])  # 928
    # print("----------------------------------------------------------------------------------")

    pd.set_option('display.width', 1000)
    pd.set_option('display.max_row', 100)
    pd.set_option('display.max_columns', 17)

    #print(df.loc[(df['type'] == 'Car') & (df['occluded'] == 3) & (df['file_num']>135) ])
    #print(df.loc[(df['type'] == 'Car') & (df['occluded'] == 3)])
    # 8,8,25,33,36,36,49,49,59,61,85,92,101,102,110,112,135,146,146,161,161,165
    # 7458,7464,7475,7480,7480

    #print(df.loc[(df['file_num']==7480)])
    # 51849  ~ 51864
    # 51849 51854

    image = cv2.imread('./data/007480.png', cv2.IMREAD_COLOR)


    # left ,top ,right, bottom = occDrowbox(37)
    # cv2.rectangle(image, ( left , top ) , (right , bottom ), (255 ,  0 ,  0) , 3)

    left ,top ,right, bottom = occDrowbox(51849)
    cv2.rectangle(image, ( left , top ) , (right , bottom ), (0 ,  0 ,  255) , 3)
    left, top, right, bottom = occDrowbox(51850)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51851)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51852)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51853)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51854)
    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 3)
    left, top, right, bottom = occDrowbox(51855)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51856)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51857)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51858)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51859)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51860)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51861)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51862)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51863)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    left, top, right, bottom = occDrowbox(51864)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)

    cv2.imshow("image", image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def occDrowbox(no):
    bbox_left = df.loc[[no], ['bbox_left']]
    bbox_top = df.loc[[no], ['bbox_top']]
    bbox_right = df.loc[[no], ['bbox_right']]
    bbox_bottom = df.loc[[no], ['bbox_bottom']]
    left = bbox_left.iloc[0][0]
    top = bbox_top.iloc[0][0]
    right = bbox_right.iloc[0][0]
    bottom = bbox_bottom.iloc[0][0]
    return  int(left) , int(top) , int(right) , int(bottom)


if __name__ == '__main__':
    test_columns = ['type', 'truncated', 'occluded', 'alpha', 'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
                    'dimensions_height', 'dimensions_width', 'dimensions_length', 'location_x', 'location_y',
                    'location_z',
                    'rotation_y', 'file_num']

    # 1번만 실행 (label 데이터 전체를 csv 파일로 저장)+ 파일명 라벨 추가
    #saveCSV(7481)

    # # csv 데이터 읽기
    # # 출력 세팅
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_row', 100)
    pd.set_option('display.max_columns', 17)

    df = pd.read_csv("data/tot_label.csv")

    # 데이터 요약
    #print(df.describe())
    
    # 상위 10개 데이터 출력 
    #print(df.head())

    #print(df)

    # 객체 종류와 갯수 출력

    #print(df['type'].value_counts())

    # Car 만 분석
    #car_analy(df)
    # 나머지

    # occluded 분석
    #occludedChk(df)


    # car , van , truck
    # occluded (3 제외하고 검색)

    # 산점도 - type 에 따라 마커의 모양을 다르게
    # 2D
    #type_ScatterPlot01()
    #type_ScatterPlot02()
    type_ScatterPlot03()
    # 3D
    #type_ScatterPlot_3D_01()
    # type_ScatterPlot_3D_02()
    #Scatter_plot_on_polar_axis()

    #type_label()

