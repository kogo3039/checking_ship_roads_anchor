
import pandas as pd
import pickle
import os
import folium
from scipy import stats
import numpy as np
from numpy import genfromtxt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

plt.rcParams.update({'figure.dpi': '60'})
plt.rcParams.update({'figure.figsize': [8, 6]})
plt.rcParams.update({'font.size': '15'})
plt.rcParams.update({'font.family': 'AppleGothic'})

def avg_residual(x1, x2):


    residual = x1 - x2
    residual = np.power(residual, 2)
    residual = np.sum(residual) / 19

    print(f"잔차의 평균: {residual}")



# 정규화
def normal_graph(x1, x2):

    plt.figure(figsize=(15, 10))
    plt.title('Standard Normal Distribution(test3)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.plot(x1, norm.pdf(x1, loc=0, scale=1), label='기준')
    plt.plot(x2, norm.pdf(x2, loc=0, scale=1), label='테스트')
    plt.legend()
    plt.show()

# 맵 만들기
def mmsi_location_map(filename, mmsi, path):

    seaMap = folium.Map(location=[35.95, 127.7], zoom_start=9)
    df = pd.read_csv(filename)
    for i in range(df.shape[0]):
        location = [df.loc[i, 'latitude'], df.loc[i,'longitude']]
        marker = folium.Circle(location=location,
                               fill='blue')
        marker.add_to(seaMap)
    seaMap.save(os.path.join(path, f"loadMap_{mmsi}.html"))

#공분산
def covariance(csv_file):

    matrix = genfromtxt(csv_file, delimiter=',', skip_header=1)

    sog = matrix[:, 3]
    time = matrix[:, 5] % 1000
    matrix = matrix[:, 1:3]

    start = matrix[-1]
    x_s = matrix - start
    sw_cov = x_s.T @ x_s

    return start, sw_cov, matrix, sog, time


# 베이즈 확률
def bayes_theorem(x, s, cov):

    inv_cov = np.linalg.inv(cov)

    g = - (x-s) @ inv_cov @ (x-s).T

    return g


#lda 선형 변형식
def Linear_tranformation(cov_lsts, finish_point_lsts):


    SW = np.zeros([2,2])
    for cov in cov_lsts:
        SW += cov
    means = np.array(finish_point_lsts)
    mu = np.mean(means, axis=0)
    means_mu = means - mu
    SB = means_mu.T @ means_mu

    W = np.linalg.inv(SW) @ SB

    return W

#pca 차원 축소 기법
def principal_component_analysis(matrix):

    w, v = np.linalg.eig(matrix)
    v0 = [v[0,0], v[1,0]]
    v1 = [v[0,1], v[1,1]]

    if w[0] > w[1]:
        pc = np.array(v0)
    else:
        pc = np.array(v1)

    return pc

def degree_one_point(path):

    with open("pickle/lda_model.pickle", 'rb') as lf:
        W = pickle.load(lf)

    with open("pickle/pca_model.pickle", 'rb') as lf:
        pca = pickle.load(lf)

    df = pd.read_csv(path)
    x = df.loc[:,['latitude', 'longitude']]
    x = np.array(x)
    M = W @ pca
    x = M @ x.T

    mini = np.min(x)
    maxi = np.max(x)
    # mu = np.mean(x)
    # sigma = np.std(x)
    nor_X = (x - mini) / (maxi - mini)

    return nor_X


def test_road_map(filename):

    df = pd.read_csv(filename)
    sns.scatterplot(data=df, x='sog', y='mapping_values')
    plt.show()



if __name__ == "__main__":

    path = 'one_dim/'
    filenames = os.listdir(path)
    if '.DS_Store' in filenames:
        filenames.remove('.DS_Store')
    filenames = sorted(filenames)

    # 기준 데이터 파일
    path1 = os.path.join(path, filenames[0])

    # 테스트 데이터 파일
    path2 = os.path.join(path, filenames[1])

    # 기준 일차 노드
    criteria = pd.read_csv(path1)
    criteria = criteria.loc[:, 'sample']
    criteria = np.array(criteria)

    # 테스트 일차 노드
    x_test = degree_one_point(path2)
    # print(x_test)
    # exit()

    # 일차원 축소 맵
    #test_road_map(os.path.join(path, filenames[1]))

    # 정규화 비교
    normal_graph(criteria, x_test)

    # 잔차의 평균
    avg_residual(criteria, x_test)













