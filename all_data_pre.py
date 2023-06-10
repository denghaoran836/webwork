# coding=UTF-8
import sys
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import xlrd
import xlwt
from xlutils.copy import copy

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
file_seq=["00","01","02","04","05","06","07","08","09"]

def read_excel_xls(path):
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    for i in range(0, worksheet.nrows):
        for j in range(0, worksheet.ncols):
            print(worksheet.cell_value(i, j), "\t", end="")  # 逐行逐列读取数据
        print()
 
 
def write_excel_xls_append(path, value):#excel追写内容
    index = len(value)  # 获取需要写入数据的行数
    workbook = xlrd.open_workbook(path)  # 打开工作簿
    sheets = workbook.sheet_names()  # 获取工作簿中的所有表格
    worksheet = workbook.sheet_by_name(sheets[0])  # 获取工作簿中所有表格中的的第一个表格
    rows_old = worksheet.nrows  # 获取表格中已存在的数据的行数
    new_workbook = copy(workbook)  # 将xlrd对象拷贝转化为xlwt对象
    new_worksheet = new_workbook.get_sheet(0)  # 获取转化后工作簿中的第一个表格
    for i in range(0, index):
        new_worksheet.write(rows_old, i, value[i])  # 追加写入数据，注意是从i+rows_old行开始写入
    new_workbook.save(path)  # 保存工作簿
    print("xls格式表格【追加】写入数据成功！")
    read_excel_xls(path)

def read_loam_gt(time_info,filepath):
    #loam_time=[],loam_tx=[],loam_ty=[],loam_tz=[],loam_qx=[],loam_qy=[],loam_qz=[],loam_tw=[]
    #时间戳 x y z q.x q.y q.z q.w 
    loam_time=[]
    loam_tx=[]
    loam_ty=[]
    loam_tz=[]
    loam_qx=[]
    loam_qy=[]
    loam_qz=[]
    loam_qw=[]
    with open(filepath) as f:
        for line in f.readlines():
            temp=line.split()
            if round(float(temp[0]),4) in time_info:
                loam_time.append(round(float(temp[0]),4))
                loam_tx.append(round(float(temp[1]),5))
                loam_ty.append(round(float(temp[2]),5))
                loam_tz.append(round(float(temp[3]),5))
                loam_qx.append(round(float(temp[4]),5))
                loam_qy.append(round(float(temp[5]),5))
                loam_qz.append(round(float(temp[6]),5))
                loam_qw.append(round(float(temp[7]),5))
    return loam_time,loam_tx,loam_ty,loam_tz,loam_qx,loam_qy,loam_qz,loam_qw

def read_gt(time_info,filepath):
    #时间戳 x y z q.x q.y q.z q.w 点云的数量
    time=[]
    gt_tx=[]
    gt_ty=[]
    gt_tz=[]
    gt_qx=[]
    gt_qy=[]
    gt_qz=[]
    gt_qw=[]
    points_num=[]
    with open(filepath) as f:
        for line in f.readlines():
            temp=line.split()#or round(float(temp[0])+0.0001,4) in time_info or round(float(temp[0])-0.0001,4) in time_info
            if round(float(temp[0]),4) in time_info  :
                time.append(round(float(temp[0]),4))
                gt_tx.append(round(float(temp[1]),5))
                gt_ty.append(round(float(temp[2]),5))
                gt_tz.append(round(float(temp[3]),5))
                gt_qx.append(round(float(temp[4]),5))
                gt_qy.append(round(float(temp[5]),5))
                gt_qz.append(round(float(temp[6]),5))
                gt_qw.append(round(float(temp[7]),5))
                points_num.append(int(temp[8]))

    return time,gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw,points_num

def read_info(filepath):
    #时间戳信息
    time_info=[]
    with open(filepath) as f:
        for line in f.readlines():
            temp=line.split()
            time_info.append(round(float(temp[0]),4))
    return time_info

def read_info2(time,filepath):
    #时间戳信息，局部角点信息，局部平面点信息，未采样的角点信息，降采样的角点信息，未采样的平面点信息，降采样的平面点信息，角点匹配结果1，2，平面点匹配结果1，2；
    time_info=[]
    mapcor_num=[]
    mapsuf_num=[]
    allcor_num=[]
    samcor_num=[]
    allsuf_num=[]
    samsuf_num=[]
    cnum1=[]
    cnum2=[]
    snum1=[]
    snum2=[]
    with open(filepath) as f:
        for line in f.readlines():
            temp=line.split()
            if round(float(temp[0]),4) in time:
                time_info.append(round(float(temp[0]),4))
                mapcor_num.append(int(temp[1]))
                mapsuf_num.append(int(temp[2]))
                allcor_num.append(int(temp[3]))
                samcor_num.append(int(temp[4]))
                allsuf_num.append(int(temp[5]))
                samsuf_num.append(int(temp[6]))
                cnum1.append(int(temp[7]))
                cnum2.append(int(temp[8]))
                snum1.append(int(temp[9]))
                snum2.append(int(temp[10]))
    return time_info,mapcor_num,mapsuf_num,allcor_num,samcor_num,allsuf_num,samsuf_num,cnum1,cnum2,snum1,snum2
    
#list1/list2
def com_rate(list1,list2):
    list3=[]
    if len(list1)!=len(list2):
        print('com_rate: list length not equal')
        print(len(list1),len(list2))
        sys.exit(1)
    for i in range(len(list1)):
        if list1[i]==0:
            a=0
        else:
            a=float(list1[i])/float(list2[i])
        list3.append(round(a,5))
    return list3

#误差计算，使用平方差形式
def com_error(tx1,ty1,tz1,tx2,ty2,tz2,time1,time2):
    if len(tx1)==len(ty1) and len(ty1)==len(tz1) and len(tz1)==len(tx2) and len(tx2)==len(ty2) and len(ty2)==len(tz2) and len(ty2)==len(time1) and len(time1)==len(time2):
        list1=[]
        for i in range(len(tx1)):
            if time1[i]==time2[i]:
                error=((tx1[i]-tx2[i])**2+(ty1[i]-ty2[i])**2+(tz1[i]-tz2[i])**2)**0.5
                list1.append(error)
            else:
                print("com_error: time error")
                sys.exit(1)
        return list1
    else:
        print('com_error: list length not equal')
        sys.exit(1)

#四元数转化为角度
def quat_to_degree(qx,qy,qz,qw):
    deg_x=[]
    deg_y=[]
    deg_z=[]
    if len(qx)==len(qy) and len(qy)==len(qz) and len(qz)==len(qw):
        for i in range(len(qx)):
            quat=[qx[i],qy[i],qz[i],qw[i]]
            r = R.from_quat(quat)
            Rm = r.as_matrix()
            euler0 = r.as_euler('xyz', degrees=True)
            deg_x.append(round(euler0[0],5))
            deg_y.append(round(euler0[1],5))
            deg_z.append(round(euler0[2],5))
        return deg_x,deg_y,deg_z
    else:
        print('quat_to_degree: list length not equal')
        sys.exit(1)

#变化矩阵计算
def com_matrix(tx,ty,tz,qx,qy,qz,qw):
    quat=[qx,qy,qz,qw]
    r = R.from_quat(quat)
    Rm = r.as_matrix()
    T=np.array([[Rm[0,0],Rm[0,1],Rm[0,2],tx],
                [Rm[1,0],Rm[1,1],Rm[1,2],ty],
                [Rm[2,0],Rm[2,1],Rm[2,2],tz]
                ,[0,0,0,1]])
    return T

#将转化矩阵中的旋转矩阵提取出来，并转为四元数
def matrix_to_quat(T):
    T_rot=([[T[0,0],T[0,1],T[0,2]],
            [T[1,0],T[1,1],T[1,2]],
            [T[2,0],T[2,1],T[2,2]]])
    r3 = R.from_matrix(T_rot)
    qua = r3.as_quat()
    return round(qua[0],5),round(qua[1],5),round(qua[2],5),round(qua[3],5)


#将坐标系转化到第一时刻
def trans_to_start(tx,ty,tz,qx,qy,qz,qw):
    T0=com_matrix(tx[0],ty[0],tz[0],qx[0],qy[0],qz[0],qw[0])
    if len(tx)==len(ty) and len(ty)==len(tz) and  len(tz)==len(qx) and len(qx)==len(qy) and len(qy)==len(qz) and len(qz)==len(qw):
        for i in range(len(tx)):
            T_now=com_matrix(tx[i],ty[i],tz[i],qx[i],qy[i],qz[i],qw[i])
            T_new=np.dot(T0.T,T_now)

            qx[i],qy[i],qz[i],qw[i]=matrix_to_quat(T_new)
            tx[i]=round(T_new[0,3],5)
            ty[i]=round(T_new[1,3],5)
            tz[i]=round(T_new[2,3],5)
        return tx,ty,tz,qx,qy,qz,qw
    else:
        print('trans_to_start: list length not equal')
        sys.exit(1)

#计算两个位姿之间的距离
def com_dis(tx,ty,tz):
    distance=[0.0]
    if len(tx)==len(ty) and len(ty)==len(tz):
        for i in range (len(tx)-1):
            dis= ((tx[i+1]-tx[i])**2+(ty[i+1]-ty[i])**2+(tz[i+1]-tz[i])**2)**0.5
            distance.append(round(dis,5))
        if len(distance)==len(tx):
            return distance
        else:
            print('distance length wrong')
            sys.exit(1)
    else:
        print('com_dis: list length not equal')
        sys.exit(1)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  

#计算两个位姿之间的速度
def com_vel(dis,time):
    vel=[0.0]
    if len(dis)==len(time):
        for i in range(len(dis)-1):
            vel_now=(dis[i+1])/(time[i+1]-time[i])
            vel.append(round(vel_now,5))
        if len(vel)==len(dis):
            return vel
        else:
            print('vel length wrong')
            sys.exit(1)
    else:
        print('com_vel: list length not equal')
        sys.exit(1) 

#计算位姿到原点的累计距离
def com_all_dis(dis):
    all_dis=[]
    dis_now=0.0
    for i in range(len(dis)):
        dis_now+=dis[i]
        all_dis.append(dis_now)
    if len(dis)==len(all_dis):
        return all_dis
    else:
        print('all_dis length wrong')
        sys.exit(1)

#前一时刻误差计算
def pre_error(res):
    res.pop()
    res.insert(0,0.0)
    return res

#两个列表对应位置数据相加
def sum_list(list1,list2):
    list3=[]
    if len(list1)==len(list2):
        for i in range(len(list1)):
            num=list1[i]+list2[i]
            list3.append(num)
    return list3

#计算两个位姿之间的时间差
def sub_time(time):
    new_time=[0.0]
    for i in range(len(time)-1):
        sub=time[i+1]-time[i]
        new_time.append(sub)
    return new_time

#计算一段时间内的均值,方差
def com_avge(time_list,data):
    if len(time_list)!=len(data):
        print('com_avge length wrong')
        sys.exit(1)
    delt_time=1.0
    avge_list=[]
    cov_list=[]
    time_now=time_list[0]
    data_now=[]
    count=0
    for i in range(len(time_list)):
        if time_list[i]<time_now+delt_time:
            data_now.append(data[i])
            count+=1
        else:
            avge_now=[]
            cov_now=[]
            avge=sum(data_now)/len(data_now)
            cov=sum([(x - avge) ** 2 for x in data_now]) / len(data_now)
            avge_now.append(round(avge,5))
            cov_now.append(round(cov,5))
            avge_list=avge_list+count*avge_now
            cov_list=cov_list+count*cov_now
            data_now.clear()
            time_now=time_list[i]
            data_now.append(data[i])
            count=1
    avge_now=[]
    cov_now=[]
    avge=sum(data_now)/len(data_now)
    cov=sum([(x - avge) ** 2 for x in data_now]) / len(data_now)
    avge_now.append(round(avge,5))
    cov_now.append(round(cov,5))
    avge_list=avge_list+count*avge_now
    cov_list=cov_list+count*cov_now
    if len(avge_list)!=len(cov_list) or len(avge_list)!=len(data):
        print('com_avge: result length wrong')
        print(len(cov_list),len(data))
        sys.exit(1)
    return avge_list,cov_list

#计算变化率
def com_change_rate(data):
    rate=[0.0]
    for i in range(len(data)-1):
        rate_now=(data[i+1]-data[i])/data[i]
        rate.append(round(rate_now,5))
    if len(rate)!=len(data):
        print('com_change_rate: result length wrong')
        sys.exit(1)
    return rate



#残差可视化
def data_analysis(list1,type_name):
    fig=plt.figure(figsize=(15,5))
    plt.plot(range(len(list1)), list1, 'b', label=type_name+' residual')
    fig.savefig(seq+'-'+type_name+' residual')
    plt.legend()
    plt.title(type_name+' residual')
    plt.show()
    

def com_dis2st(x,y,z):
    dist=[]
    for i in range(len(x)):
        dis=(x[i]**2+y[i]**2+z[i]**2)**0.5
        dist.append(dis)
    return dist

#计算两个位姿之间的旋转误差
def com_res_rot(qx1,qy1,qz1,qw1,qx2,qy2,qz2,qw2):
    res=[]
    T_rot=([1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0])
    if len(qx1)==len(qy1) and len(qy1)==len(qz1) and len(qz1)==len(qx2) and len(qx2)==len(qy2) and len(qy2)==len(qz2) and len(qz1)==len(qw1) and len(qw1)==len(qw2):
        for i in range(len(qx1)):
            quat1=[qx1[i],qy1[i],qz1[i],qw1[i]]
            r1 = R.from_quat(quat1)
            Rm1 = r1.as_matrix()
            quat2=[qx2[i],qy2[i],qz2[i],qw2[i]]
            r2 = R.from_quat(quat2)
            Rm2 = r2.as_matrix()
            Rm3=Rm1*Rm2.T-T_rot
            num=(Rm3[0,0]**2+Rm3[0,1]**2+Rm3[0,2]**2+
                 Rm3[1,0]**2+Rm3[1,1]**2+Rm3[1,2]**2+
                 Rm3[2,0]**2+Rm3[2,1]**2+Rm3[2,2]**2)**0.5
            res.append(180*num/math.pi)
        return res
    else:
        print("com_res_rot: list length not equal")

# view 3D 
def view_trajectory(x1,y1,z1,x2,y2,z2):
    fig=plt.figure()
    ax1=plt.axes(projection='3d')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.scatter(x1,y1,z1,c='Blue') #绘制散点图
    ax1.plot3D(x1,y1,z1,'gray')#
    ax1.scatter(x2,y2,z2,c='Red') #绘制散点图
    ax1.plot3D(x2,y2,z2,'black')
    plt.show()
    fig.savefig(seq+'-trajectory.png')


def trans_yichang(y):
    for i in range(len(y)):
        if abs(y[i][0])>1000 and i>0:
            y[i][0]=y[i-1][0]

def rotate_yichang(y):
    for i in range(len(y)):
        if abs(y[i][0])>500 and i>0:
            y[i][0]=y[i-1][0]               

#多元线性回归模型构建
def trans_mul_LR(x_list,y_list,type_name):
    print(type_name+'--- mul_LinearRegression')
    
    x=np.concatenate((x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5],x_list[6],x_list[7]), axis=0)
    y=np.concatenate((y_list[0],y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],y_list[6],y_list[7]), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    '''
    x_train=np.concatenate((x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5],x_list[6],x_list[7]), axis=0)
    x_test=np.concatenate((x_list[8],x_list[9]), axis=0)
    y_train=np.concatenate((y_list[0],y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],y_list[6],y_list[7]), axis=0)
    y_test=np.concatenate((y_list[8],y_list[9]), axis=0)
    '''
    lr = LR()
    lr.fit(x_train, y_train)
    np.set_printoptions(suppress=True)
    np.set_printoptions(precision=4)
    coef_list=lr.coef_.tolist()[0]
    print(type_name,lr.coef_)

    y_pred = lr.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    plt.figure(figsize=(15,5))
    plt.tick_params(labelsize=20,width=1)
   
    plt.plot(range(len(y_test)), y_test, 'r', label='真实值')
    plt.plot(range(len(y_test)), y_pred, 'b', label='预测值')
    plt.legend(fontsize=20)
    plt.title(type_name+'误差真实值和预测值的波形图',size=25,weight='bold')
    plt.tight_layout()
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.tick_params(labelsize=20,width=1)

    plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--')
    plt.xlabel('真实值',size=20,weight='bold')
    plt.ylabel('预测值',size=20,weight='bold')
    plt.title(type_name+'误差真实值和预测值的散点图',size=20,weight='bold')
    plt.tight_layout()
    plt.show()
    print(type_name+' MSE:',MSE)
    print(type_name+' RMSE:',RMSE)
    for i in range(len(x_list)):
        print(file_seq[i],"--predict")
        y_predict=lr.predict(x_list[i])
        MSE = metrics.mean_squared_error(y_list[i], y_predict)
        RMSE = np.sqrt(metrics.mean_squared_error(y_list[i], y_predict))
        print(' MSE:',MSE)
        print(' RMSE:',RMSE)
    print('---')
    print(' ')

def decision_tree(x_list,y_list,type_name):
    print(type_name+'---')
    print(type_name+'--- decision_tree')
    x=np.concatenate((x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5],x_list[6],x_list[7]), axis=0)
    y=np.concatenate((y_list[0],y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],y_list[6],y_list[7]), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    tree_model = DecisionTreeRegressor()
    tree_model.fit(x_train, y_train)
    y_pred = tree_model.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(type_name+' MSE:',MSE)
    print(type_name+' RMSE:',RMSE)
    print('---')
    print(' ')
    plt.figure(figsize=(15,5))
    plt.tick_params(labelsize=20,width=1)
   
    plt.plot(range(len(y_test)), y_test, 'r', label='真实值')
    plt.plot(range(len(y_test)), y_pred, 'b', label='预测值')
    plt.legend(fontsize=20)
    plt.title(type_name+'误差真实值和预测值的比较图',size=25,weight='bold')
    plt.tight_layout()
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.tick_params(labelsize=20,width=1)

    plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--')
    plt.xlabel('真实值',size=20,weight='bold')
    plt.ylabel('预测值',size=20,weight='bold')
    plt.title(type_name+'误差真实值和预测值的散点图',size=20,weight='bold')
    plt.tight_layout()
    plt.show()
    for i in range(len(x_list)):
        print(file_seq[i],"--predict")
        y_predict=tree_model.predict(x_list[i])
        MSE = metrics.mean_squared_error(y_list[i], y_predict)
        RMSE = np.sqrt(metrics.mean_squared_error(y_list[i], y_predict))
        print(' MSE:',MSE)
        print(' RMSE:',RMSE)



def poly(x_list,y_list,type_name):
    print(type_name+'---')
    print(type_name+'--- poly')
    poly_reg  = PolynomialFeatures(degree=2)
    x=np.concatenate((x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5],x_list[6],x_list[7]), axis=0)
    y=np.concatenate((y_list[0],y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],y_list[6],y_list[7]), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    x_poly_train = poly_reg.fit_transform(x_train)
    x_poly_test=poly_reg.fit_transform(x_test)
    lin_reg = LR()
    lin_reg.fit(x_poly_train, y_train)
    y_pred = lin_reg.predict(x_poly_test)
    if type_name=="trans":
        trans_yichang(y_pred)
    else:
        rotate_yichang(y_pred)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(type_name+' MSE:',MSE)
    print(type_name+' RMSE:',RMSE)
    print('---')
    print(' ')
    plt.figure(figsize=(15,5))
    plt.tick_params(labelsize=20,width=1)
   
    plt.plot(range(len(y_test)), y_test, 'r', label='真实值')
    plt.plot(range(len(y_test)), y_pred, 'b', label='预测值')
    plt.legend(fontsize=20)
    plt.title(type_name+'误差真实值和预测值的波形图',size=25,weight='bold')
    plt.tight_layout()
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.tick_params(labelsize=20,width=1)

    plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--')
    plt.xlabel('真实值',size=20,weight='bold')
    plt.ylabel('预测值',size=20,weight='bold')
    plt.title(type_name+'误差真实值和预测值的散点图',size=20,weight='bold')
    plt.tight_layout()
    plt.show()
   # plt.savefig('poly-'+type_name+seq+'-Scatter'+'.png')
    plt.show()
    for i in range(len(x_list)):
        print(file_seq[i],"--predict")
        x_poly_test_new=poly_reg.fit_transform(x_list[i])
        y_predict=lin_reg.predict(x_poly_test_new)
        if type_name=="trans":
            trans_yichang(y_predict)
        else:
            rotate_yichang(y_predict)
        MSE = metrics.mean_squared_error(y_list[i], y_predict)
        RMSE = np.sqrt(metrics.mean_squared_error(y_list[i], y_predict))
        print(' MSE:',MSE)
        print(' RMSE:',RMSE)

def random_forest(x_list,y_list,type_name):
    feat_labels=["time_cha","all_time","distance","vel","all_dis","cnum1","snum1","mapcor_num_new","samcor_num_new","samsuf_num_new",#pre_res_deg,pre_res_trans,
                    "cor_rate","suf_rate","sample_cor_rate","sample_suf_rate","points_num_new","mapsuf_num_new",
                    "cor_match_rate1","suf_match_rate1","cor_match_rate3","suf_match_rate3","cormap_now_rate","sufmap_now_rate",
                    "suf_change_rate","cor_change_rate"]
    print(type_name+'--- random_forest')
    x=np.concatenate((x_list[0],x_list[1],x_list[2],x_list[3],x_list[4],x_list[5],x_list[6],x_list[7]), axis=0)
    y=np.concatenate((y_list[0],y_list[1],y_list[2],y_list[3],y_list[4],y_list[5],y_list[6],y_list[7]), axis=0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
    regressor = RandomForestRegressor(n_estimators=200, random_state=0)
    regressor.fit(x_train, y_train.ravel())
    y_pred = regressor.predict(x_test)
    MSE = metrics.mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    importances = regressor.feature_importances_
    print(importances)

    indices = np.argsort(importances)[::-1]

    for f in range(X_train.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 60,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.title('Feature Importance')
    plt.bar(range(X_train.shape[1]),
            importances[indices],
            align='center')

    plt.xticks(range(X_train.shape[1]),
            feat_labels[indices], rotation=90)
    plt.xlim([-1, X_train.shape[1]])
    plt.tight_layout()
    #plt.savefig('images/04_09.png', dpi=300)
    plt.show()

    print(type_name+' MSE:',MSE)
    print(type_name+' RMSE:',RMSE)
    print('---')
    print(' ')
    plt.figure(figsize=(15,5))
    plt.tick_params(labelsize=20,width=1)
   
    plt.plot(range(len(y_test)), y_test, 'r', label='真实值')
    plt.plot(range(len(y_test)), y_pred, 'b', label='预测值')
    plt.legend(fontsize=20)
    plt.title(type_name+'误差真实值和预测值的比较图',size=25,weight='bold')
    plt.tight_layout()
    plt.show()

    plt.scatter(y_test, y_pred)
    plt.tick_params(labelsize=20,width=1)

    plt.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--')
    plt.xlabel('真实值',size=20,weight='bold')
    plt.ylabel('预测值',size=20,weight='bold')
    plt.title(type_name+'误差真实值和预测值的散点图',size=20,weight='bold')
    plt.tight_layout()
    plt.show()
    for i in range(len(x_list)):
        print(file_seq[i],"--predict")
        y_predict=regressor.predict(x_list[i])
        MSE = metrics.mean_squared_error(y_list[i], y_predict)
        RMSE = np.sqrt(metrics.mean_squared_error(y_list[i], y_predict))
        print(' MSE:',MSE)
        print(' RMSE:',RMSE)

def read_data(seq):
    #时间戳信息
    time_info=read_info('/home/lab/Documents/point_cloud/point_info-'+seq+'.txt')

    #时间戳 x y z q.x q.y q.z q.w 点云的数量
    time,gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw,points_num=read_gt(time_info,'/home/lab/Documents/point_cloud/gt_'+seq+'.txt')
    #时间戳 x y z q.x q.y q.z q.w 
    loam_time,loam_tx,loam_ty,loam_tz,loam_qx,loam_qy,loam_qz,loam_qw=read_loam_gt(time,'/home/lab/Documents/point_cloud/aloam_gt-'+seq+'.txt')
    #时间戳信息，局部角点信息，局部平面点信息，未采样的角点信息，降采样的角点信息，未采样的平面点信息，降采样的平面点信息，角点匹配结果1，2，平面点匹配结果1，2；
    time_info,mapcor_num,mapsuf_num,allcor_num,samcor_num,allsuf_num,samsuf_num,cnum1,cnum2,snum1,snum2=read_info2(time,'/home/lab/Documents/point_cloud/point_info-'+seq+'.txt')

   
    #view_trajectory(gt_tx,gt_ty,gt_tz,loam_tx,loam_ty,loam_tz)
    #将坐标系转化到第一时刻
    gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw=trans_to_start(gt_tx,gt_ty,gt_tz,gt_qx,gt_qy,gt_qz,gt_qw)
    #view_trajectory(gt_tx,gt_ty,gt_tz,loam_tx,loam_ty,loam_tz)
    #Loam计算结果四元数转化为角度
    loam_dx,loam_dy,loam_dz=quat_to_degree(loam_qx,loam_qy,loam_qz,loam_qw)

    #groundtruth四元数转化为角度
    gt_dx,gt_dy,gt_dz=quat_to_degree(gt_qx,gt_qy,gt_qz,gt_qw)


    #角度误差计算
    res_deg=com_error(loam_dx,loam_dy,loam_dz,gt_dx,gt_dy,gt_dz,loam_time,time)
    #data_analysis(res_deg,"degree")
    res_rot=com_res_rot(gt_qx,gt_qy,gt_qz,gt_qw,loam_qx,loam_qy,loam_qz,loam_qw)
    #data_analysis(res_rot,"rotate")
    #位置误差计算x,y,z
    res_trans=com_error(loam_tx,loam_ty,loam_tz,gt_tx,gt_ty,gt_tz,loam_time,time)
    #data_analysis(res_trans,"trans")
    #两个位姿之间的时间差
    time_cha=sub_time(time)
    all_time=com_all_dis(time_cha)
    #两个位姿之间的距离
    distance=com_dis(loam_tx,loam_ty,loam_tz)

    #两个位姿之间的速度
    vel=com_vel(distance,time)

    #位姿到原点的距离
    all_dis=com_dis2st(loam_tx,loam_ty,loam_tz)

    #前一时刻误差
    pre_res_deg=pre_error(res_deg)
    pre_res_trans=pre_error(res_trans)

    #特征点比率
    cor_rate=com_rate(allcor_num,points_num)
    suf_rate=com_rate(allsuf_num,points_num)

    #采样率
    sample_cor_rate=com_rate(samcor_num,allcor_num)
    sample_suf_rate=com_rate(samsuf_num,allsuf_num)

    #匹配率
    cor_match_rate1=com_rate(cnum1,samcor_num)
    cor_match_rate2=com_rate(cnum2,samcor_num)
    cor_match_rate3=com_rate(cnum1,mapcor_num)
    cor_match_rate4=com_rate(cnum2,mapcor_num)
    suf_match_rate1=com_rate(snum1,samsuf_num)
    suf_match_rate2=com_rate(snum2,samsuf_num)
    suf_match_rate3=com_rate(cnum1,mapsuf_num)
    suf_match_rate4=com_rate(cnum2,mapsuf_num)

    #一段时间内的速度均值，方差
    vel_avge,vel_cov=com_avge(time,vel)

    #一段时间内的速距离均值，方差
    dis_avge,dis_cov=com_avge(time,distance)

    #一段时间内的角点均值，方差
    cor_avge,cor_cov=com_avge(time,samcor_num)

    #一段时间内的平面点均值，方差
    suf_avge,suf_cov=com_avge(time,samsuf_num)

    #计算点云数量变化率
    suf_change_rate=com_change_rate(samsuf_num)
    cor_change_rate=com_change_rate(samcor_num)
     
    cormap_now_rate=com_rate(mapcor_num,points_num)
    sufmap_now_rate=com_rate(mapsuf_num,points_num)
    #位移信息训练数据准备
    mapcor_num_new= [item / 4 for item in mapcor_num]
    mapsuf_num_new=[item / 4 for item in mapsuf_num]
    points_num_new=[item / 4 for item in points_num]
    samcor_num_new=[item / 4 for item in samcor_num]
    samsuf_num_new=[item / 4 for item in samsuf_num]
    trans_x=np.vstack((time_cha,all_time,distance,vel,all_dis,cnum1,snum1,mapcor_num_new,samcor_num_new,samsuf_num_new,#pre_res_deg,pre_res_trans,
                    cor_rate,suf_rate,sample_cor_rate,sample_suf_rate,points_num_new,mapsuf_num_new,
                    cor_match_rate1,suf_match_rate1,cor_match_rate3,suf_match_rate3,cormap_now_rate,sufmap_now_rate,
                    suf_change_rate,cor_change_rate)).T
    trans_y=np.vstack(res_trans)

    deg_x=np.vstack((time_cha,all_time,distance,vel,all_dis,#pre_res_trans,pre_res_deg,
                        cor_rate,suf_rate,sample_cor_rate,sample_suf_rate,
                        cor_match_rate1,cor_match_rate3,
                        suf_match_rate1,suf_match_rate3,
                        vel_avge,vel_cov,dis_avge,dis_cov,cor_avge,cor_cov,suf_avge,suf_cov,
                        suf_change_rate,cor_change_rate)).T 
    #deg_y=np.vstack(res_deg) 
    deg_y=np.vstack(res_rot)
    return trans_x,trans_y,deg_x,deg_y
# 点云数量变化情况
# 点云数量变化情况


trans_x00,trans_y00,deg_x00,deg_y00=read_data("00")
trans_x01,trans_y01,deg_x01,deg_y01=read_data("01")
trans_x02,trans_y02,deg_x02,deg_y02=read_data("02")
trans_x04,trans_y04,deg_x04,deg_y04=read_data("04")
trans_x05,trans_y05,deg_x05,deg_y05=read_data("05")
trans_x06,trans_y06,deg_x06,deg_y06=read_data("06")
trans_x07,trans_y07,deg_x07,deg_y07=read_data("07")
trans_x08,trans_y08,deg_x08,deg_y08=read_data("08")
trans_x09,trans_y09,deg_x09,deg_y09=read_data("09")
trans_x10,trans_y10,deg_x10,deg_y10=read_data("10")

transx_list=[trans_x00,trans_x01,trans_x02,trans_x04,trans_x05,trans_x06,trans_x07,trans_x09]
transy_list=[trans_y00,trans_y01,trans_y02,trans_y04,trans_y05,trans_y06,trans_y07,trans_y09]
degx_list=[deg_x00,deg_x01,deg_x02,deg_x04,deg_x05,deg_x06,deg_x07,deg_x09]
degy_list=[deg_y00,deg_y01,deg_y02,deg_y04,deg_y05,deg_y06,deg_y07,deg_y09]
'''
transx_list=[trans_x00,trans_x01,trans_x02,trans_x04,trans_x05,trans_x06,trans_x07,trans_x08,trans_x09]
transy_list=[trans_y00,trans_y01,trans_y02,trans_y04,trans_y05,trans_y06,trans_y07,trans_y08,trans_y09]
degx_list=[deg_x00,deg_x01,deg_x02,deg_x04,deg_x05,deg_x06,deg_x07,deg_x08,deg_x09]
degy_list=[deg_y00,deg_y01,deg_y02,deg_y04,deg_y05,deg_y06,deg_y07,deg_y08,deg_y09]
'''
#trans_mul_LR(transx_list,transy_list,"平移")
#trans_mul_LR(degx_list,degy_list,"旋转")
#decision_tree(transx_list,transy_list,"平移")
#decision_tree(degx_list,degy_list,"旋转")
#poly(transx_list,transy_list,"平移")
#poly(degx_list,degy_list,"旋转")

random_forest(transx_list,transy_list,"平移")
random_forest(degx_list,degy_list,"旋转")


