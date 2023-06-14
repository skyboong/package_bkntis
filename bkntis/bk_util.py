# bk_util.py
# 자주 사용하는 나에게는 유용한 기능을 모음
# BK Choi
# 2022.1.14(Fr)

import datetime
import functools
import os
import re
import string
import sys
from collections import Counter, defaultdict

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import cm
from pandas import DataFrame, Series
from PyPDF2 import PdfFileMerger, PdfFileWriter
from scipy import stats

sys.path.append('/Users/bk/Dropbox/bkmodule2019/')


# ..............................
def deco1(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        print( f"\n>>> {func.__name__}( )")
        return func(*args, **kwargs)
    return wrap

# --------------------------
#      Graph Utility
# --------------------------

# 한글 세팅
def graph_font(name='NanumGothic') -> str:
    """matplotlib의 폰트 세팅 
    name : 폰트 이름, 'NanumGothic'
    2020.4.17
    """
    matplotlib.rc('font', family=name)

def find_hex_color_from_cmap(x, colormap_name=None):
    """
    지정한 colormap에 0 ~ 1.0 사이 값을 입력하면, 16진수 형태로 컬러값을 돌려줌
    2022.4.11
    """
    if colormap_name is None:
        colormap_name = 'tab20c'

    colormap1 = cm.get_cmap(colormap_name)
    color_rgb = colormap1(x) # ex [0.1, 0.2, 0.3]
    color_hex = mcolor.to_hex(color_rgb)
    return f"{color_hex}"

def make_pairs_ex(df, colname, sep=';') -> DataFrame:
    """
    Update : 2021.9.4(Sat), 2021.11.11(Th), 2022.2.9(Thr)

    DataFrame의 컬럼(colname)에서 데이터 입력 받아서,
    구분자로 구분한후,
    이들 사이를 에지로 연결한 데이터 프레임을 리턴함

    Parameters
    ----------


    Returns
    -------


    Examples
    --------
    df=DataFrame ( [['apples;banana',10],
                ['apples;banana',10],
                ['bananas;grape',20],
                ['grapes;peach;bananas',30]],
                 columns=['c1','c2'])

    make_pair_from_seris_string(df, 'c1',';')

    """

    dflist = []
    for index, each in df.iterrows():
        # 2021.10.12 - float type 대비
        w0 = f"{each[colname]}"
        w = w0.split(sep)
        # TODO 2021.11.11 revamp (1) 공백 제거해 줌
        w2 = [ wi for wi in w if wi.strip() != '']
        # print(w)
        if len(w2)>1:
            df_temp = make_pair_from_list(w2)
            dflist.append(df_temp)

    df2 = pd.concat(dflist)
    df3 = df2.groupby(['source', 'target'])['weight'].agg(['sum'])
    df4 = df3.reset_index()
    df5 = df4.rename(columns={'sum': 'weight'})
    # df5
    return df5




def make_dic(df, col_key, col_value):
    """데이타프레임 입력받아서  키컬럼, 밸류컬럼 지정해 주면, 사전 생성해서 리턴
    2022.5.1
    """
    dic_a = df[[col_key, col_value]]
    dic_b = dic_a.set_index(col_key)
    dic_c = dic_b[col_value].to_dict()
    return dic_c



def make_pair_from_list(lista:list):
    """
    update : 2021.8.30

    ['w1','w2','w3'] 가 있을때

    w1-w2
    w1-w3
    w2-w3

    로 만들어 주기

    Parameters
    -----------
    lista : 입력데이터

    Examples
    ---------
    make_part_list(['w1','w2','w3'])

    """
    temp = []
    for i, eachi in enumerate(lista):
        for j, eachj in enumerate(lista):
            if i < j:
                temp.append([eachi, eachj])
    df = DataFrame(temp, columns=['source', 'target'])
    df['weight'] = 1
    return df


def fun_cagr(v_0, v_n, *, n, delta=1, option=1):
    """
     parameters
     ----------
     v_0 : 시작 값
     v_n : 끝  값
     n    :  기간 (예, 연차)

     option : 1. v_0 값이 0 일 경우 delta 값을 입력해 주기
              2.                 None 값을 입력해 주기

     return
     ------
     compound annual growth rate(CAGR)

     example
     -------
     fun_cagr(102,733,19)
     10.937697402661794
    """

    #if delta is None :
    #return 100.*((v_n/v_0)**(1./n)-1.0)
    #else:
    #if delta is not None:
    # 0으로 나눠지는 것 피하기 위해서 ...

    # if v_0 == 0 or v_n == 0 :
    if v_0 == 0 : # 2022.9.15
        if option == 1:
            v_0 += delta
            v_n += delta
        else:
            return None

    # return 100.*((v_n/v_0)**(1./n)-1.0)
    return 100 * (np.power(v_n/v_0, 1/n) - 1 )


#############################################

def bk_join(x, sep=' ') -> str:
    """
    2022.1.17
    컬럼에 해당하는 내용을 연결하여 새로운 컬럼을 생성함

    parameters
    ----------
    x   : 시컨스 아규먼트
    sep : 구분자, 디폴트는 스페이스

    return
    ------
    연결된 문자열 리턴함
    """

    temp=[]
    for each in x :
        if pd.isnull(each):
            #print(each)
            pass
        else:
            temp.append(f"{each}")
    # 만약에 요소가 하나도 없으면 워닝 출력 !
    a=len(temp)
    if a == 0 :
        print(f'warning ! : len(temp)={a}')
    return ' '.join(temp)


def find_year_info(data=[], year=[]):
    """처음 등장하는 해, 제일 나중에 등장하는 해, 평균해
    :param data: 값 리스트
    :param year: 연도 정보 리스트
    :return:

    sample
    ------
    find_year_info([0,200,300],[2010,2012,2014])

    2022.2.5
    """
    dic1 = {}
    # 값이 0 이상이면서 가장작은 년도 찾기
    df1 = DataFrame({'data':data,'year':year})
    df2 = df1[df1['data']>0]
    ymin = df2['year'].min()
    dic1['ymin']=ymin
    '''
    for each, y1 in zip(data,year):
        if each > 0 :
            dic1['ymin'] = y1
            break
    '''
    ymax = df2['year'].max()
    dic1['ymax'] = ymax

    temp=[]
    for each, y1 in zip(data, year):
        # print(type(each), 'each=', each)
        # print(type(y1), 'y1=', y1)
        if each > 0 :
            # print("... each=", each, " y1=", y1)
            # dic1['ymax']= y1
            # temp.extend( each * [y1] )
            for ii in range(int(each)):
                # each가 float 일때 에러나는 문제 해결하기 위해서 정수로 변경
                temp.append(y1)
    mean1 = np.mean(temp)
    q1 = np.quantile(temp, 0.25)
    q3 = np.quantile(temp, 0.75)
    dic1['yq1'] = q1
    dic1['ymean'] = mean1
    dic1['yq3'] = q3
    return dic1


def make_pearson(df=None,
                 n=3,
                 level=0,
                 delta=1,
                 with_year=False,
                 column_year=None,
                 column_group=None) -> DataFrame:
    """
    2021.8.12
    2021.8.21 df 멀티인덱스일 경우 레벨 지정해주기 : level
    2022.1.28 cagr 추가
    2022.2.5
    2022.4.16
    데이터 입력 받아서, 상관 계수 값 리턴해 줌

    parameters
    ----------
    n : 최근년도 기준

    returns
    -------
    DataFrame

    Example
    --------
    df = DataFrame([[1,2,3],[1,3,1]], columns=[2000,2001,2002], index=['row1','row2'])
    make_pearson(df,n=2)
    """
    # 입력 타입이 B인 경우
    if (column_year is not None) and (column_group is not None):
        df2=df.groupby([column_group,column_year])[column_year].count().unstack(fill_value=0)
        df = df2
    dic_list = []
    # xdata = np.array(list(range(len(df.columns))))
    xdata = df.columns.get_level_values(level).values
    # print('xdata=', xdata)
    # print(type(xdata))
    i = 0
    # year_difference = df.columns[-1] - df.columns[0]
    year_difference = xdata[-1] - xdata[0]
    #print(year_difference)
    for index, each1 in df.iterrows():
        each1 = each.tolist()
        # print("each1=", each1)
        dict1 = {}
        #corr = scipy.stats.pearsonr(xdata, df.iloc[i, :])
        #corr = stats.pearsonr(xdata, df.iloc[i, :])
        corr = stats.pearsonr(xdata, each1.tolist())
        i += 1
        v_n = each1[df.columns[-1]]
        v_0 = each1[df.columns[0]]
       # v_n = each1[]
        cagr2 = fun_cagr(v_0, v_n, n=year_difference, delta=delta)
        #print(cagr2)
        dict1['cagr'] = cagr2
        dict1['word'] = index
        dict1['r'] = corr[0]
        dict1['p-value'] = corr[1]
        dict1['count'] = each1.sum()
        dict1['new'] = each1[-1 * n:].sum()
        # print(each[-1*n:])
        dict_y = find_year_info(data=each1.tolist(), year=xdata)
        dict1.update(dict_y)
        dic_list.append(dict1)
    df_pearson = DataFrame(dic_list)
    df_pearson['r_n'] = df_pearson['new'] / df_pearson['count']
    df_pearson.set_index('word', inplace=True)
    df2 = df_pearson.dropna(subset=['r'])
    if with_year == True:
        dfr = pd.concat([df, df2], axis=1)
    else:
        dfr = df2
    dfr2 = dfr.reset_index()
    return dfr2.rename(columns={'index' : 'group'})



def make_pearson_v2(df=None,
                 n=3,
                 level=0,
                 delta=1,
                 with_year=False,
                 column_year=None,
                 column_group=None) -> DataFrame:
    """
    2021.8.12
    2021.8.21 df 멀티인덱스일 경우 레벨 지정해주기 : level
    2022.1.28 cagr 추가
    2022.2.5
    2022.4.16
    2022.5.2
    데이터 입력 받아서, 상관 계수 값 리턴해 줌

    parameters
    ----------
    n : 최근년도 기준

    returns
    -------
    DataFrame

    Example
    --------
    df = DataFrame([[1,2,3],[1,3,1]], columns=[2000,2001,2002], index=['row1','row2'])
    make_pearson(df,n=2)
    """
    # 입력 타입이 B인 경우
    if (column_year is not None) and (column_group is not None):
        df2=df.groupby([column_group,column_year])[column_year].count().unstack(fill_value=0)
        df = df2
    dic_list = []
    # xdata = np.array(list(range(len(df.columns))))
    xdata = df.columns.get_level_values(level).values

    # 컬럼이 문자열이때 정수형으로 바꿔주기 2022.5.2
    if isinstance(xdata[0], str):
        temp1 = [int(each) for each in xdata]
        xdata = temp1

    # print('xdata=', xdata)
    # print(type(xdata))
    i = 0
    # year_difference = df.columns[-1] - df.columns[0]
    year_difference = xdata[-1] - xdata[0]
    #print(year_difference)
    for index, each in df.iterrows():
        each1 = each.tolist()
        # print("each1=", each1)
        dict1 = {}
        #corr = scipy.stats.pearsonr(xdata, df.iloc[i, :])
        #corr = stats.pearsonr(xdata, df.iloc[i, :])
        corr = stats.pearsonr(xdata, each1)
        i += 1
        v_n = each1[-1]
        v_0 = each1[0]
       # v_n = each1[]
        cagr2 = fun_cagr(v_0, v_n, n=year_difference, delta=delta)
        #print(cagr2)
        dict1['cagr'] = cagr2
        dict1['word'] = index
        dict1['r'] = corr[0]
        dict1['p-value'] = corr[1]
        dict1['count'] = sum(each1)
        dict1['new'] = sum(each1[-1 * n:])
        # print(each[-1*n:])
        # dict_y = find_year_info(data=each1, year=xdata)
        # dict1.update(dict_y)
        dic_list.append(dict1)
    df_pearson = DataFrame(dic_list)
    df_pearson['r_n'] = df_pearson['new'] / df_pearson['count']
    df_pearson.set_index('word', inplace=True)
    df2 = df_pearson.dropna(subset=['r'])
    if with_year == True:
        dfr = pd.concat([df, df2], axis=1)
    else:
        dfr = df2
    dfr2 = dfr.reset_index()
    return dfr2.rename(columns={'index' : 'group'})


def make_pearson_v3(df=None,
                 n=3,
                 level=0,
                 delta=1,
                 with_year=False,
                 column_year=None,
                 column_group=None,
                    option=1) -> DataFrame:
    """
    2021.8.12
    2021.8.21 df 멀티인덱스일 경우 레벨 지정해주기 : level
    2022.1.28 cagr 추가
    2022.2.5
    2022.4.16
    2022.5.2
    데이터 입력 받아서, 상관 계수 값 리턴해 줌


    parameters
    ----------
    df : 입력형태는  컬럼에는 년도정보가 들어 있어야 함.
    n : 최근년도 기준

    returns
    -------
    DataFrame

    Example
    --------
    df = DataFrame([[1,2,3],[1,3,1]], columns=[2000,2001,2002], index=['row1','row2'])
    make_pearson(df,n=2)
    """
    # 입력 타입이 B인 경우
    if (column_year is not None) and (column_group is not None):
        df2=df.groupby([column_group,column_year])[column_year].count().unstack(fill_value=0)
        df = df2
    dic_list = []
    # xdata = np.array(list(range(len(df.columns))))
    xdata = df.columns.get_level_values(level).values

    # 컬럼이 문자열이때 정수형으로 바꿔주기 2022.5.2
    if isinstance(xdata[0], str):
        temp1 = [int(each) for each in xdata]
        xdata = temp1

    # print('xdata=', xdata)
    # print(type(xdata))
    i = 0
    # year_difference = df.columns[-1] - df.columns[0]
    year_difference = xdata[-1] - xdata[0]
    #print(year_difference)
    for index, each in df.iterrows():
        each1 = each.tolist()
        # print("each1=", each1)
        dict1 = {}
        #corr = scipy.stats.pearsonr(xdata, df.iloc[i, :])
        #corr = stats.pearsonr(xdata, df.iloc[i, :])
        corr = stats.pearsonr(xdata, each1)
        i += 1
        v_n = each1[-1]
        v_0 = each1[0]

        # 2022.9.15 수정
        if v_0 == 0 :
            # 0 보다 큰 숫자 찾기
            index_x = 0
            for index_v0, each_v0 in enumerate(each1):
                if each_v0 > 0:
                    index_x = index_v0
                    break
            v_0 = each1[index_x]
            year_difference = xdata[-1] - xdata[index_x]
            print(f"initial v_0 == 0, so changed year_difference to {year_difference}")



        # v_n = each1[]
        cagr2 = fun_cagr(v_0, v_n, n=year_difference, delta=delta, option=option)
        #print(cagr2)
        dict1['cagr'] = cagr2
        dict1['year_diff'] = year_difference
        dict1['word'] = index
        dict1['r'] = corr[0]
        dict1['p-value'] = corr[1]
        dict1['count'] = sum(each1)
        dict1['n'] = n # 2022.9.2  added
        dict1['new'] = sum(each1[-1 * n:])
        # print(each[-1*n:])
        # dict_y = find_year_info(data=each1, year=xdata)
        # dict1.update(dict_y)
        dic_list.append(dict1)
    df_pearson = DataFrame(dic_list)
    df_pearson['r_n'] = df_pearson['new'] / df_pearson['count']
    df_pearson.set_index('word', inplace=True)
    df2 = df_pearson.dropna(subset=['r'])
    if with_year == True:
        dfr = pd.concat([df, df2], axis=1)
    else:
        dfr = df2
    dfr2 = dfr.reset_index()
    return dfr2.rename(columns={'index' : 'group'})

def make_pearson_old(df=None, n=3,
                 level=0,
                 delta=1,
                 with_year=False,
                 column_year=None,
                 column_group=None) -> DataFrame:
    """
    2021.8.12
    2021.8.21 df 멀티인덱스일 경우 레벨 지정해주기 : level
    2022.1.28 cagr 추가
    2022.2.5
    데이터 입력받아서, 상관계수값 리턴해 줌

    parameters
    ----------
    n : 최근년도 기준

    returns
    -------
    DataFrame

    Example
    --------
    df = DataFrame([[1,2,3],[1,3,1]], columns=[2000,2001,2002], index=['row1','row2'])
    make_pearson(df,n=2)

    """
    # 입력 타입이 B인 경우
    if (column_year is not None) and (column_group is not None):
        df2=df.groupby([column_group,column_year])[column_year].count().unstack(fill_value=0)
        df = df2

    dic_list = []
    # xdata = np.array(list(range(len(df.columns))))
    xdata = df.columns.get_level_values(level).values

    i = 0
    year_difference = df.columns[-1] - df.columns[0]
    #print(year_difference)
    for index, each in df.iterrows():
        dict1 = {}
        #corr = scipy.stats.pearsonr(xdata, df.iloc[i, :])
        corr = stats.pearsonr(xdata, df.iloc[i, :])
        i += 1
        v_n = each[df.columns[-1]]
        v_0 = each[df.columns[0]]
        cagr2 = fun_cagr(v_0,v_n,
                         n=year_difference, delta=delta)

        #print(cagr2)
        dict1['cagr']=cagr2
        dict1['word'] = index
        dict1['r'] = corr[0]
        dict1['p-value'] = corr[1]
        dict1['count'] = each.sum()
        dict1['new'] = each[-1 * n:].sum()
        # print(each[-1*n:])
        dict1.update( find_year_info(data=each.tolist(), year=df.columns.tolist()))
        dic_list.append(dict1)

    df_pearson = DataFrame(dic_list)
    df_pearson['r_n'] = df_pearson['new'] / df_pearson['count']
    df_pearson.set_index('word', inplace=True)
    df2 = df_pearson.dropna(subset=['r'])

    if with_year == True:
        dfr =pd.concat([df, df2], axis=1)
    else:
        dfr = df2

    dfr2 = dfr.reset_index()
    return dfr2.rename(columns={'index':'group'})













    



def find_method_in(obj, **kwargs):
    """
    객체의 메소드 찾기
    2020.3.24(화) 
    
    parameters
    ----------
    obj : 객체
    **kwargs :  'show'가 True이면 프린트 출력, False 이면 출력 않함 
                 '_' 가 True 이면, __메소드 출력, False 이면 그냥 메소드 출력 
    
    returns
    -------
    list : method 
    
    example
    -------
    a = 'abc'
    mlist = find_method_in(a,_=False, show=True)  
    """
    methodlist=[]
    n = 1
    for i, attr_name in enumerate(dir(obj)):
        x = getattr(obj, attr_name)     # a 객체에서 attr_name 이름속을 지닌것들의 값을 찾아냄. 
        if callable(x):                    # 찾아낸 값이 callable 한 지 검토하여, callable들만 추출
            if attr_name.startswith('_') == kwargs.get('_',False): # 이중에서 _ 가 없는 것들만 추출함 
                if kwargs.get('show', False) == True :
                    #print(i, n, '\t', attr_name,'\t', x)
                    print(n, attr_name)
                    n +=1
                methodlist.append(attr_name)
    return methodlist 



def show_info(date1=None):
    """
    데이트타임 객체들의 정보를 보여주기 위해서 임시로 만듦
    
    2020.1.30(?)

    parameters
    ----------
    datetime.datetime 

    returns
    -------
    print()

    """
    #print(f">>> type({type(date1)})")
    temp =[]
    for each in dir(date1):
        temp.append(each)
            
    keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
    
    for each in keys:
        if each in temp:
            #print(each)
            print(f"{each}:", eval(f"date1.{each}"))


def make_new_name(name, ext):
    """
    name : 파일이름 앞
    ext  : 확장자 이름
    """
    n1 = datetime.datetime.today().strftime("%Y_%m%d_%H%M_%S")
    return f"{name}_{n1}.{ext}"

# make_new_name('name', 'json')

def make_fixedwidth_string(txt='', width=1, sep='<br>'):
    """
    고정된 폭으로 변경해 줌
    2022.9.17
    """
    a1 = len(txt) // width
    a2 = len(txt) % width
    temp = []

    p1 = 0
    if a1>0: # 몫이 0 보다 크면,
        for i in range(a1):
            p2 = (i + 1) * width
            temp.append(txt[p1:p2])
            p1 = p2
        temp.append(txt[p2:])
        return f"{sep}".join(temp)
    else:
        return txt


def find_ipc_code_single(ptn=None, file_ipc_code=None, col='코드') -> DataFrame:
    """
    한개의 ptn 입력 받아서 ipc 코드 정보 찾고, 데이터 프레임으로 리턴해 주기
    """
    print(f"* pattern={ptn}, sheet_name = {ptn[0]}")
    path, ext = os.path.splitext(file_ipc_code)
    if ext.lower() in ['.xlsx', '.xlx']:
        df = pd.read_excel(file_ipc_code, sheet_name=ptn[0])
    elif ext.lower() in ['.pkl', '.pickle']:
        df = pd.read_pickle(file_ipc_code)[ptn[0]]
    else:
        return None

    df1 = df[df[col].str.contains(ptn, case=False, na=False)]
    return df1


def find_ipc_code_multiple(ptns=None, file_ipc_code="./dic/IPC 분류표_'22.1월 버전.xlsx") -> DataFrame:
    """
    다수의 ptns 입력 받아서 ipc 코드 정보 찾고, 데이터 프레임으로 리턴해 주기

    아래 kipris site에서 IPC 엑셀파일 다운로드
    http://www.kipris.or.kr/kpat/remocon/srchIpc/srchIpcFrame.jsp?myConcern=&codeType=IPC

    Example
    -------
    names2 = ['F02B39/00', 'F02B37/18', 'F01D25/16', 'F01D17/16', 'F02B39/14']
    df_ipc = find_ipc_code_multiple(names2)
    print(df_ipc.head(1))
    """
    df_list = []
    for ptn in ptns:
        df_temp = find_ipc_code_single(ptn, file_ipc_code=file_ipc_code)
        #print(f"length = {len(df_temp)}")
        if len(df_temp) > 0:
            df_list.append(df_temp)
    if len(df_list)>=1:
        df = pd.concat(df_list, axis=0)
    else:
        df = DataFrame()
    return df


def merge_pdf(filenames, new_filename='new_pdf', remove=False):
    """
    pdf 파일을 입력받아서 새로운 pdf 파일을 생성함

    filenames : pdf 파일 이름
    new_filename : 새롭게 생성할 pdf 파일 이름
    remove : 기존의 pdf 파일을 삭제 여부
    """
    merger = PdfFileMerger()

    for fn in sorted(filenames):
        merger.append(fn)

    if new_filename.endswith('.pdf'):
        pass
    else:
        new_filename += '.pdf'

    # 만약에 기존에 동일한 파일이 있다면, 파일을 삭제하고 다시 만들어 줌
    if os.path.isfile(new_filename):
        os.remove(new_filename)
    merger.write(new_filename)
    merger.close()

    if remove == True:  # 삭제
        for each_file in filenames:
            os.remove(each_file)
            print(f"* {each_file} was deleted !")

    print(f"* {new_filename} was created !")


############################
#        집합 비교
############################

def compare_set(setx, sety, name1='set1', name2='set2', show=False):
    """
    2개 집합의 연산결과 보여줌
    
    Example
    -------
    compare_set({1,2,3}, {2,3,4})
    
    set1 | set2 = {1, 2, 3, 4},	size = 4
    set1 & set2 = {2, 3},	size = 2
    set1 - set2 = {1},	size = 1
    set2 - set1 = {4},	size = 1
    set1 ^ set2 = {1, 4},	size = 2

    """

    setz1 = setx | sety
    setz2 = setx & sety
    setz31 = setx - sety
    setz32 = sety - setx
    setz4 = setx ^ sety 
    
    if show == True:
        print(f"{name1} | {name2} = {setz1},\tsize = {len(setz1)}")
        print(f"{name1} & {name2} = {setz2},\tsize = {len(setz2)}")
        print(f"{name1} - {name2} = {setz31},\tsize = {len(setz31)}")
        print(f"{name2} - {name1} = {setz32},\tsize = {len(setz32)}")
        print(f"{name1} ^ {name2} = {setz4},\tsize = {len(setz4)}")
    else:
        print(f"{name1} | {name2} = size  {len(setz1):,}")
        print(f"{name1} & {name2} = size  {len(setz2):,}")
        print(f"{name1} - {name2} = size  {len(setz31):,}")
        print(f"{name2} - {name1} = size  {len(setz32):,}")
        print(f"{name1} ^ {name2} = size  {len(setz4):,}")

    return [len(setz1), len(setz2), len(setz31), len(setz32), len(setz4)]


def compare_set_kind(setx, sety, name1='set1', name2='set2', show=False, option='|') -> dict:
    """
    2개 집합의 연산결과 보여줌
    
    Example
    ------
    set1 = {1,2,3}
    set2 = {2,3,4}
    compare_set_kind(set1, set2, option='and')

    """
    match option:
        case '|'|'or':
            setz = setx | sety
        case '&'|'and': 
            setz = setx & sety
        case '-':
            setz = setx - sety
        case '^':
            setz = setx ^ sety
        case _ :
            # 디폴트는 합집합
            setz = setx | sety
    n1 = len(setz)
    if show == True:
        print(f"{name1} {option} {name2} = {n1}")
    return {name1:{name2:n1}}


def compare_setlist(setlist:list, namelist=None, options = ['|','&','-'], show=False) -> dict:
    """
    집합릭스트내의 집합사이의 합집합, 교집합, 차집합 결과 보여주기
    
    2022.10.20(목)
    """
    if namelist is None:
        namelist = [ f"Set{i+1}" for i in range(len(setlist))]

    dic_total = {}
    for each_option in options:
        dic_list = []
        for i, setx in enumerate(setlist):
            for j, sety in enumerate(setlist):
                dic1 = compare_set_kind(setx, sety, name1=namelist[i],name2=namelist[j], option=each_option, show=False)
                if each_option != '-':
                    if i <= j:
                        dic_list.append(dic1)
                    else:
                        pass
                else:
                    dic_list.append(dic1)

        df_row = dic_list_to_dataframe(dic_list)
        columns = df_row.columns
        #print(columns)
        col1 = columns[0]
        col2 = columns[1]
        col3 = columns[2]
        if show == True:
            print("#"*20)
            print(f" set operator : {each_option}")
            print('-'*20)
            print(df_row)
        df_row2 = df_row.pivot_table(index=col1, columns=col2, values=col3, fill_value='-') # aggfunc=np.sum
        dic_total[each_option] = df_row2
    return dic_total




###########################
#  Gpass 검색식 만들기 
###########################

def gpass_make_search( id_list =[], max_value=2000, prefix="gpass_") -> str:
    """
    특허고유번호리스를 입력하면 GPASS 검색을 위한 검색식을 텍스트 파일로 저장해 줌
     2000건 단위로 구분함
    
    id_list : 고유번호 리스트
    max_value : 디폴트값은 2000, 2000이상이면 다운로드 할 때 문제가 발생하여 다운로드 되지 않음
    
    Return
    ------
    검색식 리턴해 줌 


    2022.10.19(수)
    
    """
    
    size = len(id_list)
    if size < 1 :
        print("(warning) Input data is not enough for making a search expression !")
    
    n1 = size//max_value # 몫
    n2 = size%max_value  # 나머지
    
    for i in range(n1+1):

        # 초기값
        x1 = i*2000
        
        # 끝값
        if i != n1 :
            x2 = x1 + max_value
        else:
            x2 = x1 + n2 
            
        C21 = id_list[x1 : x2 ]

        s1 = ' OR '.join(C21)
        s2 = f'PN=({s1})'

        

        # s2
        # 번호헷갈려서 1부터 시작
        filename = f"{prefix}{i+1:02}_from{x1:05}_to{x2-1:05}({len(C21)}).txt"
        with open(filename, 'w') as fn:
            fn.write(s2)
            print(f"len(s1) = {len(C21)}, file saved : {filename}")
    # 인덴트 조심
    return s2 


def count_total_from_list(doublelist):
    """
    이중리스트 입력받아서 카운팅하기
    """
    c1 = Counter()
    for each in doublelist:
        c1 += Counter(each)
    return c1    



############################
#        사전 관련 작업
############################


def dic_list_to_dataframe(dic_list:list) -> DataFrame :
    """
    이중 사전리스트 입력받아서 데이타프레임으로 변경시켜줌
    
    Example
    -------
    dic_list1 = [
                 {'A': {'A': 16255, 'D':200}},
                 {'A': {'B': 16255}},
                 {'A': {'C': 22341}}
                ]
    dic_list_to_dataframe(dic_list1)
    
    2022.10.20(목)
    """

    data_row = []
    for i, each_dic in enumerate(dic_list):
        for k1, v1 in each_dic.items():  
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    data_row.append([k1, k2, v2])
            else:
                data_row.append([k1, '-', v1]) 
            
    df_row = DataFrame(data_row, columns=['node1','node2','value'])
    return df_row


def make_sum_dictionary_from_dic_list(dic_list: list) -> dict:
    """
    # 사전 리스트의 값을 합한 새로운 사전을 만드는 방법 (2022.8.29 Mond)
    """
    dic_total = defaultdict(float)

    for each_dic in dic_list:
        for k, v in each_dic.items():
            dic_total[k] += v

    return dic_total


def dictionary_sum(dic_list, exclude=[{},None], option=1) -> dict :
    """
    사전리스트에서 사전 합하기
    
    Example
    -------
    dictionary_sum([{'A':1, 'B':2}, {'A':1, 'B':3, 'C':1}], option='+')
    
    defaultdict(int, {'A': 2, 'B': 5, 'C': 1}
    
    """
    
    if isinstance(dic_list, list) :
        pass
    else:
        return dict()
    
    match option :
        case 1|'+'|'sum' :
            dic_temp = defaultdict(int)
        case 2|'set' :
            dic_temp = defaultdict(set)
        
    for each in dic_list:
        if pd.isna(each):
            print(f'null data 입니다 :{each}')
        else:
            for k, v in each.items():
                if v in exclude:
                    print(f">>> v= {v}")
                    pass
                else:
                    # 개수 업데이트
                    match option :
                        case 1|'+'|'sum':
                            dic_temp[k] += v
                        case 2|'set':
                            dic_temp[k].update(v)
                        case _ :
                            print(f"(warning) : {v}")
    return dic_temp

def dictionary_sum_to_dataframe(df=None, columns=None, options=None, suffix='(total)') -> DataFrame:
    """
    df 의 colummns 에 있는 데이터(집합 타입 또는 숫자)를 입력 받아서, 모두 합하여 주고 난 다음, 데이트프레임을 리턴함
    
    options :  집합 더하기 'set', 
               숫자 더하기 '+', 'sum'
               
    2022.11.4 
    """
    dic_temp_list = []
    columns_options = zip(columns, options)
    for each_col, each_option in columns_options: 
        dic_temp = dictionary_sum(df[each_col].tolist(), option= each_option)
        dic_temp_list.append(dic_temp)
    new_columns = [ f"{each}{suffix}" for each in columns ]
        
    df_a = DataFrame(dic_temp_list, index =new_columns).T
    # Null 데이터 일 때 빈집합을 넣어주기 2022.11.4 (이 문제로 몇 시간 지체됨)
    #df_b = df_a.fillna()  # '{}'
    return df_a




def read_dictionary4(filenames=[], sep1='<', sep2=';') -> dict:
    """
    2021.8.29
    2022.8.16
    2022.11.3

    사전파일을 불러와서 사전형식으로 리턴해줌

    parameters
    ----------

    filename : 머징파일 이름

    return
    ------
    DataFrame

    Example
    ------
    read_dictionary2(filename='dic_non_eng.dic')

    """
    dic_total = dict()
    if filenames is None:
        print(" filenames is None")
        return None 
    elif isinstance(filenames, list):
        pass
    else:
        filenames = [filenames]
        print('filenames = ', filenames)

    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as fn:
            dic_merging = fn.read()
            dic1 = read_dictionary_from_string4(dic_merging, sep1=sep1, sep2=sep2)
            dic_total.update(dic1)
    return dic_total


def read_dictionary_from_string4(contents: str, sep1='<', sep2=';', option='U'):
    """
    문단 입력받아서 사전을 만들어 주기 

    

    텍스트 입력받아서 사전 만들어줌
    Example
    -------
    read_dictionary_from_string4("apple1<apple2;apple3", option='title')

    2021.2.3
    2022.8.16
    2022.11.3     
    """
    dic1 = {}
    for each in contents.split('\n'):
        # print(each)
        if each.count('#') > 0:  # '#' exist ?
            # print(1)
            continue
        elif (each.strip() in ['', 'nan']):  # 빈줄
            # print(2)
            continue
        else:
            # print(3, each)
            match option.upper():
                case 'LOWER'|'L':
                    each1 = each.strip().lower()
                case 'UPPER'|'U':
                    each1 = each.strip().upper()
                case 'TITLE'|'T':
                    each1 = each.strip().title()
                case _ :
                    each1 = each.strip().upper()
         
            # 왼쪽 구분자로 나눔
            values = each1.split(sep1, 1)
            if len(values) <= 1 :
                print(values)
            else:
                # 대표단어 설정
                name_rep = values[0].strip()

                # 각 단어를 키 값으로 가지는 사전 생성
                for each_name in values[1].split(sep2):
                    # names가 키 값임
                    names = f"{each_name}".strip()
                    if names != '':
                        dic1.update({names: name_rep})
    # 인덴트 조심
    return dic1




def find_length(x):
    """size 측정함수"""
    if pd.isna(x):
        return 0
    elif isinstance(x, set):
        return len(x)
    else:
        print("check your data type")
        
    #print(f"x={x}, {len(x)}, {type(x)}")
    return 0


##################
# count
##################


def count_method(data, sep=';', option='int', kind=1) -> dict:
    """
    문자열을 구분자로 구분하여 개수 카운팅하여 사전으로 리턴함
    
    couting 방법 :

    kind : 1 특허건수
           2 피인용회수

    option : 'int' 정수
             'float' 분수

    return
    ------
    dict()

    Example
    -------
    출원인별 피인용수 : 분수 기준
    count_method(['출원인1;출원인2',10], sep=';', option='float', kind=2)

    """
    dic1 = {}
    len_x = 0 # 출원인 수

    #if kind == 1:  # input data : str 특허 건수 계산할 때,
    if isinstance(data, str):
        if pd.isna(data):
            return {}

        x1 = data.split(sep)
        x2 = sorted(set(x1))  # 중복된 것 제거
        x3 = [each1.strip() for each1 in x2 if each1.strip() != '']
        len_x = len(x3)

        if option in [1, 'int']:
            for each3 in x3:
                dic1[each3] = 1
        else:
            for each3 in x3:
                dic1[each3] = 1 / len(x3)
    elif isinstance(data, list|tuple):
         # input data type : list, tuple , 피인용수 계산 할 때, data 는
        # data[0] : 출원인
        # data[1] : 피인용수
    
        x = data[0] # 출원인
        citation_no = data[1]

        if pd.isna(x):
            return {}
        if pd.isna(citation_no):
            citation_no=0

        x1 = x.split(sep)
        x2 = sorted(set(x1))  # 중복된 것 제거
        x3 = [each1.strip() for each1 in x2 if each1.strip() != '']

        len_x = len(x3)

        if option in [1, 'int']:
            for each3 in x3:
                dic1[each3] = citation_no
        else:
            #print(".")
            # 배분하기
            for each3 in x3:
                dic1[each3] = citation_no / len_x

        #if (citation_no > 56) and (len_x >2) :
        #    # print(f" 출원인 : {x}, 출원인수 : {len_x}, dic1={dic1}")
        #    print(f"\noption : {option}, dic1={dic1}")
    else:
        dic1 = {}

    return dic1



def count_method_as_year_count(x):
    """
    :param x: x[0]은 사전, x[1]은 year
    :return:
    """
    dic1 = x[0]
    year = x[1]
    dic2 = {}
    for k, v in dic1.items():
        dic2[k] = {year: v}
    return dic2




# -----
# 사람 정보 불러오기
# -----
if True :
    def find_date_from_bk_style(txt):
        temp = []
        if txt.strip().startswith('#'):
            print('comment = ', txt)
            return None
        elif txt.strip().startswith('@'): 
            print('..')
            return None
        else:
            each2 = re.split(r"[:|~|@]",txt)
            each3 = [w.strip() for w in each2 if w.strip() != '']
            temp.extend(each3)
    
        if len(temp) == 4:
            columns = ['name','birth','death','comment']
            
        elif len(temp) == 3:
            columns = ['name','birth','death_y']
        else:
            print(f'check your data ! :{txt}, {temp} = {len(temp)}')
            return 
        dic = {k:v for k,v in zip(columns, temp)}
        
        def convert_date(x):
            a = re.match(r"(-?\d+)[-](\d+)[-](\d+)", x)
            if a :
                return int(a.group(1))
            else:
                return int(x)
        dic['birth_y'] = convert_date(dic['birth'])
        dic['death_y'] = convert_date(dic['death'])
        
        return dic 
        
    def find_date_from_bk_style1(txt=None):
        """
        
        example
        ----
        find_date_from_bk_style(txt = "헨리 데이빗 소로우:1817~1862@미국;시민불복종")
        
        """
        #ptn1 = r"([가-힣\w\s]+):(\d{4})~(\d{4})@([가-힣\w\s;]+)"
        greek= "ͰͱͲͳʹͶͷͺͻͼͽͿΆΈΉΊΌΎΏΐΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩΪΫάέήίΰαβγδεζηθικλμνξοπρςστυφχψωϊϋόύώϏϐϑϒϓϔϕϖϗϘϙϚϛϜϝϞϟϠϡϰϱϲϳϴϵϷϸϹϺϻϼϽϾϿ"
        korean="가-힣"
        etc = "ῆ"
        # 페리클레스(Περικλῆς):-495~-429@그리스;아테네;정치인
        ptn3 = rf"([\w{korean}{greek}{etc}\(\)\s]+):\s?([-]?\d{1,4})~([-]?\d{1,4})@([가-힣\w\s;]+)"
        ptn2 = r"(.\s?[가-힣\w\s]+):\s?([-]?\d{1,4})-(\d{1,2})-(\d{1,2})~([-]?\d{1,4})-(\d{1,2})-(\d{1,2})@([가-힣\w\s;]+)"
        ptn1 = r"([가-힣\w\s]+):([-]?\d{1,4})~([-]?\d{1,4})@([가-힣\w\s;]+)"
        
        aa1 = re.search(ptn1, txt)
        dic = {}
        if aa1 :
            print("Type A")
            aa = aa1.groups()
            #print(aa)
            dic['name_kor'] = aa[0]
            dic['birth_y'] = aa[1]
            dic['death_y'] = aa[2]
            dic['comment'] = aa[3]
            return dic
        else:
            
            aa2 = re.search(ptn2, txt)
            if aa2:
                print("Type B")
                aa = aa2.groups()
                #print(aa)
                name = aa[0]
                y1 = int(aa[1])
                m1 = int(aa[2])
                d1 = int(aa[3])
                y2 = int(aa[4])
                m2 = int(aa[5])
                d2 = int(aa[6])
                c1 = datetime.datetime(y1,m1,d1)
                c2 = datetime.datetime(y2,m2,d2)
                dic['name_kor'] = name.strip()
                dic['birth'] = c1
                dic['death'] = c2
                dic['birth_y'] = c1.year
                dic['death_y'] = c2.year
                return dic 
            else:
                
                aa3 = re.search(ptn3, txt)
                if aa3:
                    print("Type C")
                    aa = aa2.groups()
                    #print(aa)
                    name = aa[0]
                    y1 = int(aa[1])
                    
                    y2 = int(aa[2])
                    comment = aa[3]
                
                    dic['name_kor'] = name.strip()
                    dic['birth'] = y1
                    dic['death'] = y2
                    dic['birth_y'] = y1
                    dic['death_y'] = y2
                    dic['comment'] = comment 
                    return dic 
                else:
                    print(f"Type etc : {txt}")
                    return None 
                    
        return None



    def find_date_from_wiki_style(txt, ptn=r'영어:\s+([\w\s]+),'):
        """
        위키 스타일로 사람 정보 불러옴
        
        example
        -----
        txt = "월터 휘트먼(영어: Walter Whitman , 1819년 5월 31일 ~ 1892년 3월 26일)"
        find_date_from_wiki(txt)
        
        """
        txt2 = re.split('[()]', txt)
        print('txt2 = ', txt2)

        dic = {}

        aa1 = re.search(ptn,txt2[1])
        if aa1 :
            name = aa1.group(1).strip()
            #print('name=', name)
            dic['name_eng'] = name
        else:
            dic['name_eng'] = None
        
        ptn = '(\d{1,4})년\s(\d{1,2})월\s(\d{1,2})일 ~ (\d{1,4})년\s(\d{1,2})월\s(\d{1,2})일'
        aa1 = re.search(ptn,txt2[1])
        if aa1 :
            aa = aa1.groups()
            y1 = int(aa[0])
            m1 = int(aa[1])
            d1 = int(aa[2])
            y2 = int(aa[3])
            m2 = int(aa[4])
            d2 = int(aa[5])
            
            c1 = datetime.datetime(y1,m1,d1)
            c2 = datetime.datetime(y2,m2,d2)
            dic['name_kor'] = txt2[0].strip()
            dic['birth'] = c1
            dic['death'] = c2
            dic['birth_y'] = c1.year
            dic['death_y'] = c2.year
            return dic 
        else:
            return None
        

    def find_date(txt_list, exclude=['#','@']):
        """
        BK 스타일과 wiki 스타일 모두 처리하는 함수
        
        example
        ----
        txt1 = "헨리 데이빗 소로우:1817~1862@미국;시민불복종"
        txt2 = "월터 휘트먼(영어: Walter Whitman , 1819년 5월 31일 ~ 1892년 3월 26일)"
        find_date([txt1,txt2])
        
        2022.11.13(일)
        """
    
        dic_list = []
        for line in txt_list:
            #print(line)
            if line.strip().startswith(exclude[0]):
                print('comment = ', line)
            elif line.strip().startswith(exclude[1]): 
                print('@ =', line)
            else:
                dic1 = find_date_from_wiki_style(line)
                #print(dic1)
                if dic1 is None:
                    #print('...')
                    dic2 = find_date_from_bk_style(line)
                else:
                    dic2 = dic1
                dic_list.append(dic2)
        return DataFrame(dic_list)
    
    @deco1
    def text_to_history_format(txt='', ptn=None, filename=None, y1=None, y2=None) -> DataFrame :
        """
        history people 그래프 그려주는 데이터 포맷 
        """
        print(f"{__name__}")
        if filename is not None:
            with open(filename, 'r') as fn:
                txt = fn.read()
        else:
            pass 
        
        # 공백 아닌 줄 불러오기
        txt2 = [each for each in txt.split('\n') if each.strip() != '']
        temp = []
        for each in txt2:
            if each.strip().startswith('#'):
                #print('# = ', each)
                pass
            elif each.strip().startswith('@'): 
                #print('@')
                pass
            else:
                each2 = re.split(r"[:|~|@]", each)
                each3 = [w.strip() for w in each2 if w.strip() != '']
                temp.append(each3)
        df = DataFrame(temp)
        if len(df.columns) == 4:
            columns = ['name','birth','death','comment']
        else:
            columns = ['name','birth','death']
        df.columns = columns  
        
        #df = DataFrame(temp, columns=columns)
        df2 = df.sort_values(by=['birth'], ascending=[True])
        df2.index = pd.RangeIndex(len(df2.index))
        #df2
        df3 = df2 
        def convert_date(x):
            a = re.match(r"(-?\d+)[-](\d+)[-](\d+)", x)
            if a :
                return int(a.group(1))
            else:
                return int(x)
        df3['date1']=df3['birth'].map(lambda x : convert_date(x))
        df3['date2']=df3['death'].map(lambda x : convert_date(x))
        df4 = df3.sort_values(by=['date1'], ascending=[True])
        df4.index = pd.RangeIndex(len(df4.index))
        df4['name2'] =  df4['name'] + '(' + df4['date1'].astype(str) + '~' + df4['date2'].astype(str) + ')'
        
        # patten 제한 : 컬럼  name, comment 에서 검색
        
        if (ptn is not None) and (ptn != ''):
            if 'comment' in df4.columns:
                df5 = df4[df4['comment'].str.contains(pat=ptn, na=False) |
                          df4['name'].str.contains(pat=ptn, na=False) 
                         ]
            else:
                print(f"ptn={ptn}")
                con1 = df4['name'].str.contains(pat=ptn, na=False) 
                print(f"con1={con1}")
                df5 = df4[con1]        
        else:
            df5 = df4 
        
        # time 제한 
        if isinstance(y1, int) and isinstance(y2, int):
            # y1 이상, y2 이하 
            df6 = df5[(df5['date1'] >= y1) & (df5['date2'] <= y2)]
        elif isinstance(y1, int) :
            df6 = df5[(df5['date1'] >= y1)]
        elif isinstance(y2, int) :
            df6 = df5[(df5['date2'] <= y2)]
        else:
            df6 = df5
        return df6
    
    #@deco1
    def text_to_history_format_b(txt='', ptn=None, filename=None, y1=None, y2=None) -> DataFrame :
        """
        history people 그래프 그려주는 데이터 포맷 
        """
        if filename is not None:
            with open(filename, 'r') as fn:
                txt = fn.read()
        else:
            pass 
        
        # 공백 아닌 줄 불러오기
        txt2 = [each for each in txt.split('\n') if each.strip() != '']
        df3 = find_date(txt2)
        print(df3)
        
        df3['date1']=df3['birth_y'].map(lambda x : int(x))
        df3['date2']=df3['death_y'].map(lambda x : int(x))
        df3['age']=df3['date2']-df3['date1']
        
        def convert_date_span(x):
            x1=x[0]
            x2=x[1]
            if x1 == x2 :
                return f"{x1}"
            else:
                return f"{x1}~{x2}"
        df3['life_span'] = df3[['date1','date2']].apply(lambda x : convert_date_span(x), axis=1)
        df3['name2'] =  df3['name_kor'] + '(' + df3['life_span'] + ')'
        df4 = df3.sort_values(by=['date1'], ascending=[True])
        df4.index = pd.RangeIndex(len(df4.index))
        
        # patten 제한 : 컬럼  name, comment 에서 검색
        if (ptn is not None) or (ptn != ''):
            df5 = df4[df4['comment'].str.contains(ptn, na=False) |
                    df4['name_kor'].str.contains(ptn, na=False) 
                    ]
        else:
            df5 = df4 
        
        # time 제한 
        if isinstance(y1, int) and isinstance(y2, int):
            # y1 이상, y2 이하 
            df6 = df5[(df5['date1'] >= y1) & (df5['date2'] <= y2)]
        elif isinstance(y1, int) :
            df6 = df5[(df5['date1'] >= y1)]
        elif isinstance(y2, int) :
            df6 = df5[(df5['date2'] <= y2)]
        else:
            df6 = df5
        return df6






if __name__ == '__main__':
    print(f"file name : Utility File {__file__}")
   
