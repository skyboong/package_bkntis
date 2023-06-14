"""  NTIS 주제별 과제목록 추출 함수
    bk_ntis_find.py
    2022.12.7
    by B. K. Choi
"""
import time
import datetime 
import xlsxwriter
import datetime
import os
import sys
import re
import random
from collections import defaultdict
import functools
import sys
sys.path.append('/Users/bk/Dropbox/bkmodule2019/')
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import bk_excel2 as be


def deco1(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        #global global_no
        #global_no += 1
        # print( f">>> {func.__name__}, args={args}, kwargs={kwargs}")
        # print(f">>> {func.__name__}()")
        return func(*args, **kwargs)
    return wrap

codes =[
 ('분류1 표준', ['과학기술표준분류1-대', '과학기술표준분류1-중', '과학기술표준분류1-소',
               '과학기술표준분류2-대', '과학기드술표준분류2-중', '과학기술표준분류2-소', 
               '과학기술표준분류3-대', '과학기술표준분류3-중', '과학기술표준분류3-소',
               '과학기술표준_연구분야분류1', '과학기술표준_연구분야분류2', '과학기술표준_연구분야분류3' ]),
 ('분류2 중점', [ '중점과학기술분류-대', '중점과학기술분류-중', '중점과학기술분류-소' ]),
 ('분류3 적용', [ '적용분야1', '적용분야2', '적용분야3', 
               '과학기술표준_적용분야분류1', '과학기술표준_적용분야분류2', '과학기술표준_적용분야분류3']),
 ('분류4 녹색', ['녹색기술분류','녹색기술분야분류']),
 ('분류5 6T', [ '6T관련기술-대', '6T관련기술-중', '6T관련기술-소', '6T관련기술분류']),
 ('분류6 전략', ['국가전략기술'] ),

 ('사업명', ['사업명(정제)']),
 ('내역사업',['내역사업명(정제)']), 
 ('키워드', ['KW1','KW2']), 
 ('제목', ['TI(정제)']),
]
code_names1 = [
        '과학기술표준분류1-대',
        '과학기술표준분류1-중',
        '과학기술표준분류1-소',
        '과학기술표준분류2-대',
        '과학기드술표준분류2-중',
        '과학기술표준분류2-소',
        '과학기술표준분류3-대',
        '과학기술표준분류3-중',
        '과학기술표준분류3-소',
        '중점과학기술분류-대',
        '중점과학기술분류-중',
        '중점과학기술분류-소',
        '적용분야1',
        '적용분야2',
        '적용분야3',
        '녹색기술분류',
        '6T관련기술-대',
        '6T관련기술-중',
        '6T관련기술-소',
        # '국가전략기술-대',
        # '국가전략기술-중',
        # '국가전략기술-소',
        ]
code_names2 = [
        '과학기술표준_연구분야분류1',
        '과학기술표준_연구분야분류2',
        '과학기술표준_연구분야분류3',
        '과학기술표준_적용분야분류1',
        '과학기술표준_적용분야분류2',
        '과학기술표준_적용분야분류3',
        '녹색기술분야분류',
        '6T관련기술분류',
        '국가전략기술',
        '부처자체분류_대분류',
        '부처자체분류_중분류',
        '부처자체분류_소분류'
        ]
columns_info1 = ['AB', 'AB(정제)', 'AB2','AB2(정제)', 'EF', 'EF(정제)', '중분류', '중분류(싱글)', '중분류(멀티)']
columns_basic = ['과제고유번호','DP','DP(정제)', 
                 '사업명','사업명(정제)', '내역사업명', '내역사업명(정제)', 
                 'TI', 'TI(정제)',
                 'PY','FUND',
                 '과제수행기관명', '과제수행기관명(정제)', 
                 'AR(정제)','AR_sub(정제)',
                 'KW1(정제)','KW2(정제)']
columns_total = columns_basic + code_names1 + code_names2 + columns_info1 

# 사전
dic_dp = {'과학기술정보통신부': '(과기)',# 수정
            '교육부': '(교육)',
            '중소벤처기업부': '(중소)',
            '산업통상자원부': '(산업)',
            '농촌진흥청': '(농진)',
            '보건복지부': '(보건)',
            '농림축산식품부': '(농축)',
            '해양수산부': '(해수)', # 수정 
            '다부처': '(다부)',
            '국무조정실': '(국조)',
            '환경부': '(환경)',
            '국토교통부': '(국토)',
            '식품의약품안전처': '(식품)',
            '산림청': '(산림)',
            '방위사업청': '(방사)',# 수정
            '기상청': '(기상)',
            '행정안전부': '(행안)',
            '원자력안전위원회': '(원자)',
            '문화체육관광부': '(문체)', # 수정
            '기획재정부': '(기재)',
            '해양경찰청': '(해경)', # 수정
            '소방청': '(소방)',
            '질병관리청': '(질병)',
            '문화재청': '(문화)', #
            '경찰청': '(경찰)',
            '국방부': '(국방)',
            '고용노동부': '(고용)',
            '통일부': '(통일)',
            '인사혁신처': '(인사)',
            '특허청': '(특허)',
            '새만금개발청': '(새만)',
            '여성가족부': '(여가)',
            '외교부': '(외교)',
            '공정거래위원회': '(공정)',
            '법무부': '(법무)',
            '행정중심복합도시건설청': '(행도)', # 수정
            '법제처': '(법제)',
            '개인정보보호위원회': '(개인)'}

# 과제수행기관명
dic_org = { 
           "경상국립대학":"경상대학",
           "스페이스솔류션":"스페이스솔루션",
           "항공우주연구원":"한국항공우주연구원",
        }

org_company = [   '(주)',
'(주',
'(주식회사)',
'[주]',
'{주)',
'<주>',
'㈜',
'주식회사',
'주)']

global_prefix_list = [
# prefix list (2023.2.4토)
# 아래 문자가 단어 제일 앞에 등장하면 삭제하게 함

# 농업회사법인
'농업회사법인','농업법인','(농)',
# 재단법인
'(재)','(재단)','(조합)','재단법인',
# 사단법인
'(사)','(사단)','(사단법인)','사단법인','사)','[사단]',
# 사회복지법인
'사회복지법인',
# 어업회사법인
'어업회사법인',
# 영농조합법인
#영농조합법인 # 조심해서 포함시킬 것
# 의료법인
'의료법인','(의)','의)','(의료)',
# 국립대학법인
'국립대학법인',


# 주식회사
'( 주)', '(주 )','(주)','(주.','(주식회사)','(쥬)','[주]','{주)','（주）','<주>','㈜','주식회사','주)',

# 유한회사
'(유)','유)','(유한)','유한회사', '유한책임회사',



#------------
# 학교
#------------
'(학)', '(학교)',
'(학교법인원광학원)',
'(강원지역혁신플랫폼)',
'(경남지역혁신플랫폼)',
'(공동수급)',
'(광주전남지역혁신플랫폼)', '(대구경북지역혁신플랫폼)','(대전세종충남지역혁신플랫폼)','(울산경남지역혁신플랫폼)', '(충북지역혁신플랫폼)',

'(사협)',
'(영)',
'(특)','(특수법인)', '(합)', '(합자)',

]
global_prefix_list1 = [
                r'^재단법인',
                r'^사단법인',
                r'^유한회사',
                r'^법무법인',
                r'^사회복지법인',
                r'^농업회사법인',
                r'^농업법인',
                r'^어업회사법인', # 2023.2.4 added
                r'^의료법인',

                r'^\(주\)\s?농업회사법인',  # 이런 경우도 있군
                r'^주식회사\s?농업회사법인',
                r'^국립대학법인',
                r'^학교법인',
                r'^주식회사',
                r'^\(\s?주\s?\)',  # (주)
                r'^（주）',  # 겉으로 보기에는 비슷해 보여도 괄호가 아닌 문자
                r'^주\)',  # 주)
                r'^㈜',
                r'^\[주\]',
                r'^\(주식회사\)',
                r'^\(재\)',
                r'^\(재단\)',
                r'^\(사\)',
                r'^\(사단법인\)',
                r'^\(사단\)',
                r'^\[사단\]',
                r'^\(의\)',
                r'^\(의료\)',
                r'^의료\)',
                r'^\(유\)',
                r'^\(유한\)',
                r'^\(학\)',
                r'^\(학교\)']

global_suffix_list = [
# suffix list (2023.2.4토)
# 아래 문자가 등장하면 삭제하게 함
'(교)','(법인)',
#------------
# 사단법인
#------------
'(사)', '(사단법인)',

'(조합)', '(합)',
#------------
# 재단법인
#------------
'(재)', '(재단)', '[재단]', '[조합]',

# 주식회사
'[주]', '(주)', '주)', '주식회사', '㈜',

#------------
# 유한회사
#------------
'유한책임회사', '(유)', '(유한)', '유한회사',

#------------
# 책임회사
#------------
'책임회사',

#------------
# 농업회사법인
#------------
'농업회사법인', '농업법인',

#------------
# 어업회사법인
#------------
'어업회사법인', #영농조합법인 # 조심해서 포함시킬 것
               ]

global_suffix_list1 = [
                r'\(주\)$',
                r'㈜$',
                r'주식회사$',
                r'\(재\)$',
                r'\(재단\)$',
                r'\[재단\]$',
                #  r'재단$', # 한국연구재단 이 한국연구로 변경되어서 이 패턴은 제외 (2022.12.18)
                r'\(사\)$',
                r'\(사단법인\)$',
                r'\(유한\)$',
                r'유한회사$',
                r'책임회사$',
                r'기술지주회사$',
               # r'법인$',
                r'농업회사법인$',
                r'농업회사법인\s주식회사$',  # '비토바이오 농업회사법인 주식회사'
                r'주식회사\s?농업회사법인$',
                r'\(주\)\s?농업회사법인$',
                r'농업회사법인\s?\(주\)'
               ]

global_ptn_univ_list = [
    # 대학원대학 문제
    r"([가-힣\w]+대학원대학교?)$",
    r"([가-힣\w]+대학원대학교)\s?([가-힣\w]+)$",
    # 협의회 문제
    r"([\w\s·]+대학[\w\s·]+[협의회|협회])$",
    # 산학협력단 문제 : 국민대학교 산학협력단, 서울과학기술대학원 산학협력단, 연세의료원산학협력단
    r"([가-힣\w]+대학교?)\s?(\(?[가-힣\w]+\s?캠퍼스)\)?\s?([가-힣\w]+협력단)$",  # 00대학교 00캠퍼스 산학협력단
    r"([가-힣\w]+(대학원|대학교|교대|대학|대))\s?((글로컬산학|연구산학|산학|산학|산합|산업)협력단)$",
    r"^([\w\s]+대학교)\s?((글로컬|에리카|\w+)\s?(연구산학|산학|산합)협력단)", # 건국대학교 글로컬산학협력단, 제주대학교산합협력단
    r"^([\w\s]+[대학교|대학])\s?(산학\s?협력단)([\w\s_]*)",
    # 캠퍼스 문제
    r"([가-힣\wI]+대학교?)\s?\(?([가-힣\w]+캠퍼스\)?)$",  # 성균관대학교(자연과학캠퍼스), 한국폴리텍I대학서울강서캠퍼스
    # 대학교 단과대학 문제
    r"([가-힣\w]+대학교?)\s?([가-힣\w]+대학)$",
    #r"([가-힣\w]+대학교?)\s?([가-힣\w]+대학)\s?([\w\s]+)(?!병원|의료원|한의원)$",
    # 000대학교00연구소, 00대학교 00대학 00연구소, 00대학교 Human-inspired 복합지능 연구센터, '경인교육대학교학습및정서·행동문제연구소'
    #r"([가-힣\w\s]+대학교?)\s?([\w\s\-·]+)(?!병원|의료원|한의원)",
   # r"([가-힣\w\s]+대학교?)\s?([\w\s\-·]+)(?!병원|의료원|한의원)",

    # 과학기술원 : 한국, 대구경북, 울산, 광주
    r"^([\w\s]+과학기술원)(?!\s?부설|고등과학원|선박해양플랜트연구소)",
    # 한국연구원 추출
    r"(한국[\w]+연구원)(?!\s?부설)",
    r"(^[가-힣\wI\s]+대학교?)$",  # 00대학교대

    # 한국예술종합학교
    r"^([\w\s]+종합학교)[\w\s\(\)]*"
    ]

global_ptn_company_list = ['^([^농업][\w\s]+)\(주\)[\w\s]+',
                           '(.+)기업\s?부설\s?연구소',
                           ]
global_ptn_hospital_list = [
    # 병원
    r"^[\w\s]+대학교\s?(분당차병원)",  # 차의과대학교 분당차병원
    r"^([\w\s]+병원|의료원|한의원|의원)",
    #연세대학교의료원산학협력단, #연세대학교산학협력단(의료원)
    r"^([\w\s]+대학교?.+\(의료원\))"
    ]

# ------------------------------------------------------------------------









# method 2

name = '가천대학교 길병원'


# name = '가천대학교'

def find_hospital_name(name,
                       df_dic=None,
                       name_lemma='병원이름',
                       name_univ='대학명',
                       name_others='별칭',
                       kind=1,
                       show=False):
    """
    데이터프레임 df_dic 을 입력 받아서, name이 name_others에 있으면, name_lemma를 리턴함
    name_others는 ;으로 구분 되어 있음
    만약 df_dic 이 None 이면, name 리턴함
    kind 1 : 대표병원명 리턴
         2 : 대학명 리턴
    example
    find_hospital_name(name=name, df_dic=df_dic1, name_lemma='병원이름', name_others='별칭', show=True)
    """
    if show == True:
        print(f"입력 : {name}")

    if df_dic is None:
        return name

        # df_dic[name_others]에 입력한 name 있으면 True, 아니면 False

    def compare1(x, sep=';', name=''):
        if pd.isna(x):
            return False
        else:
            x2 = x.split(sep)
            x3 = [each.strip() for each in x2 if each.strip() != '']
            if name in x3:
                return True
            else:
                return False

    tf = df_dic[name_others].map(lambda x: compare1(x, sep=';', name=name))
    df_dic2 = df_dic[tf].copy()

    # 만약에 존재함다면, 레마를 리턴함
    match kind :
        case 1|'h':
            if len(df_dic2.index) > 0:
                name = df_dic2[name_lemma].tolist()
                name2 = ';'.join(name).strip()
                # print(f'>>> * {name2}')
                if show == True:
                    print(f"출력 : {name2}")
                return name2
            else:
                return name
        case 2|'u':
            if len(df_dic2.index) > 0:
                name_univ = df_dic2[name_univ].tolist()
                name2 = ';'.join(name_univ).strip()
                # print(f'>>> * {name2}')
                if show == True:
                    print(f"출력 : {name2}")
                return name2
            else:
                return name




def find_info(df, ptn,
              column_names=[], 
              col_fund='FUND', 
              col1='DP(정제)', 
              col2='사업명(정제)', 
              col_PY='PY', 
              save=False, 
              file_prefix='',
              by='u', 
              show=False,
              dir_output=r'./output/'):
    """
    df의 컬럼(column_names)에서 ptn에 해당하는 정보가 있는지를 출력함 
    col의 합계 정보를 알려줌
    
    2022.12.4 
    """
    onoff_list = []
    txt_list = []
    
    # check 
    for each in [col_fund, col1, col2, col_PY] :
        if each in df.columns:
            pass
        else:
            print(f"check your input data : column name({each}) is not exist in df ")
            return False
    match column_names:
        case 'code_all'|'all':
            #code_names_total = code_names1 + code_names2
            #column_names = code_names_total
            column_names = code_names1 + code_names2  
    
    # check : dataframe columns check 
    column_names1 = [ each for each in column_names if each in df.columns]         
    column_names = column_names1 
    printa("find_info()", kind=1)
    print('* input column names :', column_names)
    
    for i, each in enumerate(column_names):
        onoff = df[each].str.contains(ptn, na=False, flags=re.IGNORECASE)
        n_sum = onoff.sum() 
        
        txt1 = "*"*50
        txt2 = f"* (지정 컬럼 {i+1}): {each}, (과제수:{n_sum}개)"
        txt3 = ''
        txt4 = ''
        if n_sum > 0 : 
            df2 = df[onoff]
            if (col_fund is not None) and (col_fund in df2.columns):
                sum1 = df2[col_fund].sum()
                txt3 = f"* 연구비 총액 = {sum1/100000000:,.1f} 억원, 과제수 = {n_sum} 개"

            s1 = df2[each].value_counts()
            txt4 = f"{s1.to_string()}"
        txt_list.extend([txt1, txt2, txt3, txt4])    
        onoff_list.append(onoff)
    
    onoff_total = pd.concat(onoff_list, axis=1)
    print('* onoff_total.shape = ', onoff_total.shape)
    
    # 합집합 or 교집합
    if by == 'u' :
        print(f"by = {by} : union")
        con1 = onoff_total.any(axis=1)
    else:
        print (f"by ={by} : ")
        con1 = onoff_total.all(axis=1)
        
    number_project = con1.sum() # 과제수 
    txt5 = f"분류조건을 만족하는 과제수 = {number_project}"
    
    df_r1 = df[con1]
    if show : 
        print(f"* length of df_r1 is {len(df_r1)} ")
        if len(df_r1.index) == 0 :
            print("* length of df_r1 is 0 ")
            return False 
    
    # result : 
    df_table1 = df_r1.groupby([col1, col2, col_PY])[col_fund].agg('sum').unstack(fill_value=0)
    n_a = len(df_table1.index)
    # print( f" : {n_a:,}개")
    if n_a == 0 :
        print("....check your data !")
        
    df_table1b = df_table1/1e8
    
    #df_r1.groupby(['DP', '사업명'])['FUND'].agg(['sum'])
    df_table2 = df_r1.groupby([col1, col2])[col_fund].agg('sum')
    df_table2b = df_table2/1e8
    fund_total = df_table2b.sum() # 연구비 총액
    
    df_table3 = df_r1.groupby([col1, col2, col_PY])[col_fund].agg('count').unstack(fill_value=0)
    number_program = len(df_table2b.index) # 사업 개수
    
    print(f"사업수 : {number_program:,}개, 과제수 :{number_project:,}개, 투자액:{fund_total:,.1f}억원")
    if save is True:
        # 과제수 0 이상 일 때만 하기
        if number_project > 0 : 
            filename = file_prefix + f"({number_project}개).txt"
            #  error 63 : file name too long 
            if len(filename) >=20:
                filename = filename[0:20]
            
            with open( dir_output + filename, 'w') as fn:
                a = datetime.datetime.now()
                a2 = a.strftime('%Y-%m-%d %H:%M:%S')
                
                fn.write(f'분석결과 보고 (작성 : {a2})\n >>> 키워드 패턴 = {ptn}\n\n')
                fn.write('='*50)
                fn.write(f'\n<Part 1> 컬럼 조사 결과\n')
                fn.write('='*50 + '\n')
                fn.write(f"* 조사 컬럼 : {len(column_names)}개 컬럼 : {column_names}\n")
                fn.write(f"\n사업수 : {number_program:,}개, 과제수 :{number_project:,}개, 투자액:{fund_total:,.1f}억원\n")
                fn.write('\n'.join(txt_list))
                fn.write('\n\n')
                fn.write('='*50)
                fn.write(f'\n<Part 2> 부처별 사업별 연구비(억원)\n')
                fn.write('='*50)
                
                fn.write(df_table2b.to_string())
        
                fn.write('\n\n')
                fn.write('='*50)
                fn.write(f'\n<Part 3>부처별 연도별 사업별 연구비(단위 : 억원)>\n')
                fn.write('='*50 + '\n')
                fn.write(df_table1b.to_string())
                
                fn.write('\n\n')
                fn.write('='*50)
                fn.write(f'\n<Part 4> 부처별 사업별 과제수(단위 : 개)\n')
                fn.write('='*50 + '\n')
                fn.write(df_table3.to_string())

                print(f"➡ file saved successfully : {filename}")
    # 정보 출력 
    if show : 
        print("show={show}")
        for each in txt_list:
            print(each)       

    return {'con':con1, 'table1':df_table1b, 'table2':df_table2b, 'table3':df_table3}

@deco1
def find_org_info(ptn=None, 
                  df_db=None, 
                  col_target='과제수행기관명', 
                  col_fund='FUND', 
                  col_AR='AR(정제)', 
                  col_PY='PY',
                  show=True):
    """ 
    col_target(디폴트 과제행기관명)을 입력하면, col_target의 참여과제수, 연구비총액을 집계하여 리턴해 줌
    
    2022.12.11(일)
    """
    def fun1(x):
        x2 = [str(each) for each in x]
        return ';'.join(x2) 
        
    onoff1 = df_db[col_target].str.contains(ptn, na=False, case=False)
    df2 = df_db[onoff1].copy() 
    df3 = df2.groupby([col_target, col_AR])[col_fund].agg(['sum'])/1e8 
    df4 = df2.groupby([col_target, col_AR])[col_fund].agg(['count']) 
    df5 = df2.groupby([col_target, col_AR])[col_PY].apply(lambda x : fun1(x)) 
    df6 = df2.groupby([col_target, col_AR])[col_PY].agg(['mean'])
    df6 = df6.rename(columns={'mean':'평균(PY)'})
    df7 = pd.concat([df3, df4, df5, df6], axis=1)
    df8 = df7.reset_index()
    df9 = df8.rename(columns={'sum':'연구비(억원)', 
                              'count':'과제수',
                              #'AR(정제)':'지역',
                              })
    df9['평균(억원)'] = df9['연구비(억원)']/df9['과제수']
    
    if show :
        printa(f"pattern = {ptn}, input shape = {df_db.shape}, 과제수 총합 : {df9['과제수'].sum():,}, 연구비총액(억원) : {df9['연구비(억원)'].sum():,.1f} ")
    return df9

# 함수이름 넓혀주기
find_word_info = find_org_info 


def find_info_col1_col2(ptn=None, 
                  df_db=None, 
                  col_1='과제수행기관명', 
                  col_fund='FUND', 
                  col_2='과제수행기관명(정제)', 
                  col_PY='PY',
                  show=True):
    """ 
    find_org_info 기능 개선 (2022.12.20)
    ptn이 col_1에서 발견되는 것을 추출할 때, col_2와 비교해 주는 함수 
     
    2022.12.11(일)
    2022.12.20(화)
    """
    if col_1 == col_2 :
        columns = [col_1]
    else:
        columns = [col_1, col_2]
    
    def fun1(x):
        x2 = [str(each) for each in x]
        return ';'.join(x2) 
        
    onoff1 = df_db[col_1].str.contains(ptn, na=False, case=False)
    df2 = df_db[onoff1].copy() 
    df3 = df2.groupby(columns)[col_fund].agg(['sum'])/1e8 
    df4 = df2.groupby(columns)[col_fund].agg(['count']) 
    df5 = df2.groupby(columns)[col_PY].apply(lambda x : fun1(x)) 
    df6 = df2.groupby(columns)[col_PY].agg(['mean'])
    df6 = df6.rename(columns={'mean':'평균(PY)'})
    df7 = pd.concat([df3, df4, df5, df6], axis=1)
    df8 = df7.reset_index()
    df9 = df8.rename(columns={'sum':'연구비(억원)', 
                              'count':'과제수',
                              #'AR(정제)':'지역',
                              })
    df9['평균(억원)'] = df9['연구비(억원)']/df9['과제수']
    
    if show :
        printa(f"입력 : ab {df_db.shape[0]:,} ad {onoff1.sum():,} : pattern = {ptn}, 과제수 총합 : {df9['과제수'].sum():,}, 연구비총액(억원) : {df9['연구비(억원)'].sum():,.1f} ")
    return df9

def replace_project_name(x):
    """사업명, 내역사업명 정제 함수
    """
    # 소재부품기술개발  
    # 소재부품기술개발(R&D)  
    # 나노융합산업핵심기술개발 
    # 나노융합산업핵심기술개발(R&D) 
    
    # 1. None data 처리. 사업명의 널데이터는 없지만, 내역사업명 널데이터는 89개 존재함(2022.12.13)
    # 내역사업명 2014년 이후 부터
    if isinstance(x, (float, int)):
        return x 
    
    # 2. 앞뒤 제거
    x = x.strip()

    # 3. 띄어쓰기, 중간 부호 제거 
    pattern = re.compile(r'[\s.·ㆍ]+') 
    # 바이오.의료기술개발 -> 바이오의료기술개발 : 그런데 이렇게 하니 1.5 가 15가 됨 
    # 나노·소재기술개발
    # 의료데이터보호ㆍ활용기술개발
    x = re.sub(pattern, '', x)

    # --------
    # pattern
    # --------
    # 4) 내역사업명앞의 - 부호 제거. (예) -정지궤도복합위성해양탑재체개발, 정지궤도복합위성해양탑재체개발
    ptn = r'^[\-.·ㆍ]+' 
    x = re.sub(ptn, '', x)
    
    
    # 5) 사업명(부처명)(R&D)   : (예) 국가생명연구자원선진화사업(농진청)(R&D), 해양경찰현장맞춤형연구개발(오션랩)(R&D)
    ptn = r"\([\w\s,&+\-]+\)\(R&D\)$" 
    x = re.sub(ptn, '', x)  
    
    # 6) 괄호 2개 이어 나타나는 경우 : 
    # (예) 재난안전연구원기본경비(총액)(공공질서)
    # (예) 국제핵융합실험로공동개발사업(기금,R&D)(과기정통부, 산업부)
    # (예) 문화재연구소기본경비(총액인건비대상)(R&D)) <-- 입력 데이터 에러로 보이지만 이런 경우도 처리하게 해줌 
    ptn = r"\([\w\s,&+\-]+\)\([\w\s,&+\-]+\)\)?$"
    x = re.sub(ptn, '', x) 

    # 7) 내역사업명이 그냥 ( ...) 형식으로 있는 경우  : (예)  (전파위성), (방송미디어)
    ptn1 = r"^\([\w\s,]+\)$"
    temp = re.findall(ptn1, x)
    if len(temp) > 0 :
        #print(x) 
        x = re.sub(r'\(', '', x)
        x = re.sub(r'\)', '', x)
        #print(x) 
        
    # 8) 사업명(부처명) 
    # (예) 00사업(국토부)
    # (예) 한-인도네시아산업혁신연구협력사업(R&D,ODA)
    # (예) 국립기상과학원(인건비+기본경비)
    # (예) 지역농업연구기반및전략작목육성(보조/경제)
    # (예) 내역사업 : 신진연구(총연구비1.5억초과~3억이하)
    ptn = r"\([\w\s,&+\-/.~]+\)$" 
    x = re.sub(ptn, '', x) 

    # 9) 괄호로 시작하는 것을 제거하기 
    ptn1 = r"^\([\w\s,\-]+\)"# (유형1-2)중견연구
    x = re.sub(ptn1, '', x)

    
    '''
    # 사업명 뒤 접미사1 제거 - 회계정보 
    
    for each in [ '(일반,R&D)',  # 중소기업기술혁신개발(일반,R&D) 중소기업기술혁신개발(특별,R&D) 
                  '(특별,R&D)',
                  '(일반회계)(R&D)',
                  '(일반회계)',     # 보건의료인재양성지원사업(R&D)(일반회계), 보건의료인재양성지원사업(일반회계) 
                  # '보건의료인재양성지원사업(국민건강증진기금)(R&D)', '보건의료인재양성지원사업(기금)', '보건의료인재양성지원사업(일반회계)(R&D)'
                ]:
        if x.endswith(each):
            x = x.replace(each, '') 
    # 2. 사업명 뒤 접미사2 제거 - 부처명    
    for each in ['(경찰청)',
                 '(경찰청,과기정통부)',
                '(과기부)',
                '(과기정통부)',
                '(과기정통부)(R&D)',
                '(과기정통부,국토부,산업부,행안부)',
                '(과기정통부,농진청,산림청)',
                '(과기정통부,복지부)',
                '(과기정통부,복지부,산업부) ',
                '(교육부)',
                '(교육부)(R&D)',
                '(국토부,부처간협업)',
                '(기상청)',
                '(농진청,산림청)',
                '(복지부)',
                '(산업부)',
                '(질병청)',
                '(한국연구재단)',
                '(해수부)',
                '(환경부)'] :
        if x.endswith(each):
            x = x.replace(each, '')   
            
    for each in [  '(R&D)(도로)', '(R&D)(일반)']:
        if x.endswith(each):
            x = x.replace(each, '')   
    
    #ptn1 =  '(R&D)'
    
    ptn3 =  '(정보화,R&D)' # ICT진흥및혁신기반조성 , ICT진흥및혁신기반조성(정보화,R&D)
    ptn4 =  '(R&D,정보화)' # 'SW컴퓨팅산업원천기술개발, SW컴퓨팅산업원천기술개발(R&D,정보화) 
    ptn5 =  '바이오.의료기술개발' # 바이오의료기술개발'
    

    ptn = '(운영경비)'
    if x.endswith(ptn):
        x = x.replace(ptn, '')
        
    ptn = '(주요사업비)' # 한국과학기술원연구운영비지원(R&D)(주요사업비)          9.79
                      # 한국과학기술원연구운영비지원(주요사업비)   
                      # 한국전기연구원연구운영비지원(R&D)(운영경비)
    if x.endswith(ptn):
        x = x.replace(ptn, '')
        
    if x.endswith(ptn5):
        x = x.replace(ptn5, '')
    if x.endswith(ptn3):
        x = x.replace(ptn3, '')
    if x.endswith(ptn4):
        x = x.replace(ptn4, '')
        #print("x = ",x)
    #if x.endswith(ptn1):
    #    x = x.replace(ptn1, '')
    '''
    

        
        
    # 9) '000사업' 에서 '사업' 제거    
    ptn = '사업'  # 사업명 뒤에 붙은 사업은 삭제
    if x.endswith(ptn):
        x = x.replace(ptn, '')
        #print("x = ",x)
    return x


def replace_project_name_2023(x):
    """사업명, 내역사업명 정제 함수
    replace_project_name()을 개선

    2023.6.5(Monday)

    """
    # 소재부품기술개발
    # 소재부품기술개발(R&D)
    # 나노융합산업핵심기술개발
    # 나노융합산업핵심기술개발(R&D)

    # 1. None data 처리. 사업명의 널데이터는 없지만, 내역사업명 널데이터는 89개 존재함(2022.12.13)
    # 내역사업명 2014년 이후 부터
    if isinstance(x, (float, int)):
        return x

    ptn_list6 = [
                 r"\(인건비\+기본경비\)$",# "(예) 원예특작과학원(인건비+기본경비)"),
                 r'연구운영비지원\(R&D\)\(주요사업비\)$', # '(예) 한국생산기술연구원연구운영비지원(R&D)(주요사업비)'),
                 r'연구운영비지원\(주요사업비\)$', #'(예) 한국표준과학연구원연구운영비지원(주요사업비)'),
                 r'\(운영경비\)\s?$', #'(예) 세계김치연구소연구운영비지원(R&D)(운영경비)'),
                 r'연구운영비지원\s?$', #'(예) 한국생산기술연구원연구운영비지원'),
                 # (r'연구운영비지원', '연구운영비지원 이 포함된 경우'),
                 r'연구운영비지원\(0\.5\)$', #'(예) 한국과학기술원연구운영비지원(0.5)'),
                 r'\(주요사업비\)\s?$', #'(예) 국립수산진흥원(주요사업비)')
                 ]

    # 삭제할 패턴
    patterns = [
        r'[\s·ㆍ]+', # 공백문자(\s), 가운데 점(·ㆍ) 부호 제거  (예) 바이오.의료기술개발 -> 바이오의료기술개발
        r'^[\-·ㆍ]+', # 문자 처음이 마이너스 부호(-), 가운데 점(·ㆍ) 부호 제거. 내역사업명에서 나타남(예) -정지궤도복합위성해양탑재체개발, 정지궤도복합위성해양탑재체개발
        r"\([\w\s.,&+\-]+\)\([\w\s.,&+\-]+\)\([\w\s.,&+\-]+\)\)?$", # 괄호3개로 끝나는 경우 (예)치안현장맞춤형연구개발(폴리스랩)(R&D)(경찰청, 과기정통부)
        r"\([\w\s, & +\-]+\)\([\w\s, & +\-]+\)\)?$",  # 괄호2개로 끝나는 경우 (예) 국제핵융합실험로공동개발사업(기금,R&D)(과기정통부, 산업부)
        r"\(R&D\)$", # 괄호 (R&D) 로 끝나는 경우 (예) 인문사회기초연구(R&D),
        r"\([\w,]+R&D\)$", # (..., R&D) (예) 창업성장기술개발(일반,R&D)
        r"\(R&D[\w,]+\)$", # (R&D, ...) (예) 축산시험연구(R&D,책임운영)
    ] + ptn_list6

    for each_ptn in patterns:
        x = re.sub(each_ptn, '', x)

    # 추출할 패턴 1
    patterns2 = [
        r"출연기관육성지원\((\w+)\)$",
        r"\(연구회소관출연기관\)(\w+)<\w+>$",
        r"\(직할출연기관\)(\w+)$",
        r"\(직할출연기관\)(\w+)\<0.5>$",
        r"\((\w+[부청처])\)$", # 괄호속의 부처명 삭제 (예) 정책연구개발사업(농림부), 예외가 있지만, 무시할 수준임 (예) 국가연구개발사업평가(과학기술종합조정지원사업일부)
    ]

    for each_ptn in patterns2:
        txt = re.findall(each_ptn, x)
        if txt:
            x=txt[0]
        else:
            pass

    # 추출할 패턴 2
    patterns3 = [ r"^(경제|인문|산업|공공|기초)\((\w+)\)$"]
    for each_ptn in patterns3:
        txt = re.findall(each_ptn, x)
        if txt :
            x=txt[0][1]
        else:
            pass
            #print('')]


    # 내역사업명이 그냥 ( ...) 형식으로 있는 경우  : (예)  (전파위성), (방송미디어)
    ptn1 = r"^\([\w\s,]+\)$"
    temp = re.findall(ptn1, x)
    if len(temp) > 0:
        # print(x)
        x = re.sub(r'\(', '', x)
        x = re.sub(r'\)', '', x)
        # print(x)


    # 8) 사업명(부처명)
    # (예) 00사업(국토부)
    # (예) 한-인도네시아산업혁신연구협력사업(R&D,ODA)
    # (예) 국립기상과학원(인건비+기본경비)
    # (예) 지역농업연구기반및전략작목육성(보조/경제)
    # (예) 내역사업 : 신진연구(총연구비1.5억초과~3억이하)
    #ptn = r"\([\w\s,&+\-/.~]+\)$"
    #x = re.sub(ptn, '', x)

    # 9) 괄호로 시작하는 것을 제거하기
    #ptn1 = r"^\([\w\s,\-]+\)"  # (유형1-2)중견연구
    #x = re.sub(ptn1, '', x)


    # '000사업' 에서 '사업' 제거
    ptn = '사업'  # 사업명 뒤에 붙은 사업은 삭제
    if x.endswith(ptn):
        x = x.replace(ptn, '')
        # print("x = ",x)
    return x


def replace_area_name(x, option=0):
    """
    지역명 정제 함수
    * 조분평 :  지역 vs  NTIS : 연구비 주집행지역. : AR로 통일함
    * 조분평 :  기초자치단체 vs NTIS :  연구비 주집행지역에 모두 포함되어 있음
    * AR(정제) ; 서울, 경상남도, 부산광역시 수준만 저장하게 함 

    x 정보를 받아들여서, 공백으로 split 하여,
    개수가 1이면 x 를, 2 이면 x[0] 을 리턴하게 함
    
    2022.12.12(월)
    """
    
    dic_region1 = {'경상남도': '경남',
                '경상북도': '경북',
                '부산광역시': '부산',
                '울산광역시': '울산',
                '대구광역시': '대구',

                '경기도': '경기',
                '서울특별시': '서울',
                '인천광역시': '인천',
                '전라남도': '전남',
                '전라북도': '전북',
                '광주광역시': '광주',

                '충청남도': '충남',
                '충청북도': '충북',
                '대전광역시': '대전',
                '세종특별자치시': '세종',
                '제주특별자치도': '제주',
                
                '강원도': '강원',
                '기타':'기타',
                '해외':'해외'}
    
   
    dic_region2 = {'경상남도': '지방',
                '경상북도': '지방',
                '부산광역시': '지방',
                '울산광역시': '지방',
                '대구광역시': '지방',

                '경기도': '수도권',
                '서울특별시': '수도권',
                '인천광역시': '수도권',

                '전라남도': '지방',
                '전라북도': '지방',
                '광주광역시': '지방',

                '충청남도': '지방',
                '충청북도': '지방',
                '대전광역시': '대전',
                '세종특별자치시': '지방',

                '제주특별자치도': '지방',
                '강원도': '지방'}

    dic_region3 = {'경상남도': '경상권',
                '경상북도': '경상권',
                '부산광역시': '경상권',
                '울산광역시': '경상권',
                '대구광역시': '경상권',

                '경기도': '수도권',
                '서울특별시': '수도권',
                '인천광역시': '수도권',
                '전라남도': '전라권',
                '전라북도': '전라권',
                '광주광역시': '전라권',

                '충청남도': '충청권',
                '충청북도': '충청권',
                '대전광역시': '충청권',
                '세종특별자치시': '충청권',

                '제주특별자치도': '제주',
                '강원도': '강원'}
    
    dic_region_list = [dic_region1, dic_region2, dic_region3]
        
    if x is None:
        print(x) 
        return None
    elif isinstance(x, (float,int)) :
        #print(x) 
        return x
    else:
        x2 =  x.split(' ')
        n1 = len(x2)
        if n1 == 1 :
            name = x
        elif n1 == 2 :
            name = x2[0]
        else:
            name = x
        
        dic_region = dic_region_list[option]
        
        if name in dic_region.keys():
            name2 = dic_region[name]
        else:
            name2 = name
        return name2 
    
        
def replace_area_sub_name(x):
    """
    조분평 데이터인 기초자치단체에 값이 있으면 이 값을 리턴해 주고, 
    기초자치단체에 값이 없으면, X[0] 값을 리턴함
    
    2022.12.12(월)
    """
    tag = x[2]
    
    if tag == 'a1':
        return x[3] + '_' + x[1]
    else:
        if isinstance(x[0], (float, int)):
            return None
        
        x2 =  x[0].split(' ')
        n1 = len(x2)
        if n1 == 1 :
            name = x[3] + '_' + x2[0]
        elif n1 == 2 :
            name = x[3] + '_' + x2[1]
        else:
            print("warning ")
            name = None 
        
        #if name == '서울_고양시' :
        #    print(f"{name}")
        return name 
  
         
def replace_org_name_old1(x=None, show_ptn=False):
    """과제수행기관명 정제 함수
수
    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    if x is None :
        return x
    x2 = x.strip()
    x3 = re.sub('대학교$','대학', x2)
    return x3


def replace_org_name(x=None, prefix_list=None, show_ptn=False, hospital=False, kind=1):
    """과제수행기관명 정제 함수
     아래 처리 순서를 지켜주기를.
     2022.12.14(WED)


    kind : 1 이면 리턴할 때 사전에 없는 단어는 그냥 그래로 리턴함
           2 이면 리턴할 때 사전에 없는 단어는 '' 리턴함

     (Example)
     replace_org_name('한국대학교 산학협력단') -> 한국대학교
    """
    

    # None 처리
    x_raw = x
    if x is None:
        return x

    # 문자열 앞뒤 처리
    x3 = x.strip()

    # prefix 제거
    if prefix_list is None:
        prefix_list = [r'^\(\s?주\s?\)',  # (주)
                 r'^주\)',    # 주)
                 r'^㈜',
                 r'^\[주\]',  # [주]
                 r'^\(주식회사\)',
                 r'^주식회사',
                 r'^\(재\)',
                 r'^\(재단\)',
                 r'^재단법인',
                 r'^\(사\)',
                 r'^\(사단법인\)',
                 r'^\(사단\)',
                 r'^\[사단\]',
                 r'^사단법인',
                 r'^\(의\)',
                 r'^\(유\)',
                 r'^\(유한\)',
                 r'^유한회사',
                 r'^법무법인\(유한\)',
                 r'^법무법인',
                 r'^사회복지법인',
                 r'^농업회사법인\s?\(주\)',
                 r'^농업회사법인\s?㈜',
                 r'^농업회사법인\s?\(유\)',
                 r'^농업회사법인',       
                 r'^국립대학법인',
                 r'^\(학\)',
                 r'^\(학교\)']
    prefix_list1 = [

                   r'^재단법인',
                   r'^사단법인',
                   r'^유한회사',
                   r'^법무법인',
                   r'^사회복지법인',
                   r'^농업회사법인',
                   r'^의료법인',
                   r'^\(주\)\s?농업회사법인', # 이런 경우도 있군
                   r'^주식회사\s?농업회사법인',
                   r'^국립대학법인',
                   r'^학교법인',

                  r'^주식회사',] # 위치 중요함

    prefix_list2 = [r'^\(\s?주\s?\)',  # (주)
                    r'^（주）', # 겉으로 보기에는 비슷해 보여도 괄호가 아닌 문자
                   r'^주\)',  # 주)
                   r'^㈜',
                   r'^\[주\]',
                   r'^\(주식회사\)',

                   r'^\(재\)',
                   r'^\(재단\)',

                   r'^\(사\)',
                   r'^\(사단법인\)',
                   r'^\(사단\)',
                   r'^\[사단\]',

                   r'^\(의\)',
                   r'^\(의료\)',
                   r'^의료\)',

                   r'^\(유\)',
                   r'^\(유한\)',

                   r'^\(학\)',
                   r'^\(학교\)']
    # x4 = re.sub(r'|'.join(prefix_list),'', x3)

    # 1차 : 농업회사법인 주식회사 등을 반복하여 삭제함
    for each_ptn in prefix_list1:

        if re.search(each_ptn, x3):
            #print(each_ptn, x3)
            txt = re.sub(each_ptn, '', x3)
            x3 = txt.strip()
            if show_ptn == True:
                print(f"each_ptn = {each_ptn}")
    # 2차 : 특수부호가 있는 표기 제거  : (주) 등
    for each_ptn in prefix_list2:
        if re.search(each_ptn, x3):
            x4 = re.sub(each_ptn, '', x3.strip())
            #print(f"each_ptn = {each_ptn}")
            break
        else:
            x4 = x3

    #x4 = x5

    # suffix  제거 
    suffix_list = [
        r'\(주\)$',
        r'㈜$',
        r'주식회사$',
        r'\(재\)$',
        r'\(재단\)$',
        r'\[재단\]$',
      #  r'재단$', # 한국연구재단 이 한국연구로 변경되어서 이 패턴은 제외 (2022.12.18)
        r'\(사\)$',
        r'\(사단법인\)$',
        r'\(유한\)$',
        r'유한회사$',
        r'책임회사$',
        r'기술지주회사$',
        #r'법인$',
        r'농업회사법인$',
        r'농업회사법인\s주식회사$',  # '비토바이오 농업회사법인 주식회사'
        r'주식회사\s?농업회사법인$',
        r'\(주\)\s?농업회사법인$',
        r'농업회사법인\s?\(주\)'
    
    ]
    x5 = re.sub(r'|'.join(suffix_list),'', x4.strip())
    
    # prefix, suffix 처리한 다음 결과에 대해서 처리

    x6 = replace_university_name(x5, show_ptn=show_ptn, hospital=hospital)

    x7 = replace_company_name2(name=x6, ptn_list=[], show_ptn=False)

    ptn_list_hospital = ['^([\w\s]+병원|의료원|한의원|의원)']
    x8 = replace_hospital_name2(name=x7,ptn_list=ptn_list_hospital, hospital=hospital)


    # 공백제거
    x8 = x8.replace(' ','')
    x8 = x8.strip()

    #if x8 == "부천대산업협력단": # 부천대 , 부천대학,
    #    print('x8=',x8, 'x7=',x7, 'x6=', x6, 'x5=',x5,  'x4=',x4, 'x=', x)

    if x_raw == x8 :
        if kind == 1:
            return x8
        else:
            return ''
    return x8

#@deco1
def replace_org_name2(name=None,
                      prefix_list=None,
                      suffix_list=None,
                      ptn_list=None,
                      ptn_company_list=None,
                      ptn_hospital_list=None,
                      n=2,
                      show_ptn=False,
                      hospital=True,
                      hospital_df=None,
                      kind=1):
    """과제수행기관명 정제 함수 : 2023.2.4(Sat)
       name : 입력기관명
       prefix_list : 삭제할 접두어 리스트, None 이면 디폴트 패턴 리스트 사용
       suffix_list : 삭제할 접미어 리스트, None 이면 디폴트 패턴 리스트 사용
       ptn_list : 대학명 정제 패턴 리스트,
       ptn_company_list : 기업명 정제 패턴
       ptn_hospital_list : 병원명 정제 패턴
       n : 접두어, 접미어 삭제시 반복 횟수, 디폴트는 2회
       show_ptn : 어느 패턴에 의해서 정제되었는 지를 알려줌
       hospital : 병원을 독립 시킬 것인가 태그
       hospital_df : 병원 사전 데이타프레임

        kind : 1 이면 리턴할 때 이름이 수정되지 않은 기관명은 입력값 그대로 리턴함
               2 이면 리턴할 때 이름이 수정되지 않은 기관명은 '' 리턴함
    """
    #print(">>> replace_org_name2")
    # None 처리
    name_a = '서울대학교병원'
    name_a = '차의과학대학교 분당차병원'
    #if name == name_a:
    #    print(f"before = {name}")

    if name is None:
        return None
    if prefix_list is None:
        prefix_list = global_prefix_list
        #print(prefix_list)
    if suffix_list is None:
        suffix_list = global_suffix_list
    if ptn_list is None:
        ptn_list = global_ptn_univ_list
    if ptn_company_list is None:
        ptn_company_list = global_ptn_company_list
    if ptn_hospital_list is None:
        ptn_hospital_list = global_ptn_hospital_list

    # 문자열 앞뒤 처리
    name1 = name.strip()

    # 더블 스페이스 제거
    names = name1.split(' ')
    name1 = ' '.join(names)

    # prefix, suffix 처리
    name2 = remove_prefix_suffix(prefix_list=prefix_list, suffix_list=suffix_list, name=name1, n=n)
    if name == name_a:
        print(f"(prefix, suffix) before = {name1}, after={name2}")
    # 기업명 정제
    #print('2', name2)
    name3 = replace_company_name2(name2, ptn_list=ptn_company_list, show_ptn=show_ptn)
    if name == name_a:
        print(f"(기업명정제)before = {name2}, after={name3}")
    #print('3', name3, name2)
    # 대학명 정제
    name4 = replace_university_name2(name3, ptn_list=ptn_list, show_ptn=show_ptn, hospital=hospital)
    if name == name_a:
        print(f"(대학명정제)before = {name3}, after={name4}")

    # 병원명 정제
    #print(f"병원명을 정제합니다. : {ptn_hospital_list}")
    name5 = replace_hospital_name3(name4,
                                   ptn_list=ptn_hospital_list,
                                   show_ptn=show_ptn,
                                   hospital=hospital,
                                   hospital_df=hospital_df)
    if name == name_a:
        print(f"(병원명정제)before = {name4}, after={name5}")

    # 공백 제거
    name5 = name5.replace(' ', '').strip()

    if name == name5:
        if kind == 1:
            return name
        else:
            return ''
    return name5




def replace_patterns_with_user_word(pattern_list=None, repl='', name=None, show_ptn=False, kind=1):
    """prefix, suffix 정제 함수

    kind : 1 이면 리턴할 때 사전에 없는 단어는 그냥 그래로 리턴함
           2 이면 리턴할 때 사전에 없는 단어는 '' 리턴함

     (Example)
    """
    # None 처리
    if (name is None) :
        return name

    if pattern_list is None :
        pattern_list = prefix_list + suffix_list

    # 문자열 앞뒤 처리
    clean_name = name.strip()

    for i, each_ptn in enumerate(pattern_list):
        if re.search(each_ptn, clean_name):
            # print(each_ptn, x3)
            txt = re.sub(each_ptn, '', clean_name)
            clean_name = txt.strip()
            if show_ptn == True:
                print(f"pattern {i} = {each_ptn}")
    if clean_name == name:
        if kind == 1:
            return x8
        else:
            return ''
    return clean_name

@deco1
def replace_hospital_name2(name=None,
                           ptn_list=None,
                           show_ptn=False,
                           hospital=False,
                           hospital_names_ptn=r"병원|의료원|한의원"):
    """병원 이름 정제 모듈. 병원을 대학과 분리하여 집계
    hospital = True 해주면 됨"""

    # 병원, 의료원 이 포함 없으면 종료함
    if re.search(hospital_names_ptn, name):
        pass
    else:
        #print(f"1.........name = {name}")
        return name

    # 병원을 독립 시키지 않을 때, 즉 대학병원에 속한 병원은 대학명을 리턴 시킬 때,
    if hospital == False:
        #print(f"0.........name = {name}")
        name = name.replace(' ', '')
        if name in  ['분당서울대학병원','분당서울대학병원','분당서울대병원']:
            name = '서울대학'
        return name


    """ 병원명 정제 """
    if ptn_list is None:
        ptn_list = global_ptn_hospital_list
    name2 = None
    #if ptn_list == []:
    #    # 기본 패턴 적용
    #    ptn_list = ['^([\w\s]+병원|의료원|한의원|의원)']
    for i, each_ptn in enumerate(ptn_list):
        temp = re.findall(each_ptn, name.strip())
        if temp:
            if show_ptn == True:
                print(f"{i} : {each_ptn}")
            if isinstance(temp[0], tuple):
                name2 = temp[0][0]
            else:
                name2 = temp[0]
            break
    if name2 is not None:
        name = name2

    #print(f"2.........name = {name}")

    if re.search(r'대학교\s?병원$', name):
        x1 = re.sub('대학교\s?병원$', '대학병원', name)

    elif re.search(r'대학\s?병원$', name):
        x1 = re.sub('대학\s?병원$', '대학병원', name)

    elif re.search('대\s?병원$', name):
        x1 = re.sub('대\s?병원$', '대학병원', name)

    elif re.search(r'대학교?\s?치과\s?병원$', name):
        x1 = re.sub('대학교?\s?치과\s?병원$', '대학치과병원', name)

    elif re.search(r'대학교\s?의료원$', name):
        x1 = re.sub('대학교\s?의료원$','대학의료원',name)

    else:
        x1 = name
    #print(f'before : {name} after : {x1}')
    return x1


@deco1
def replace_hospital_name3(name=None,
                           ptn_list=None,
                           show_ptn=False,
                           hospital=False,
                           hospital_names_ptn=r"병원|의료원|한의원",
                           hospital_df=None):
    """병원 이름 정제 모듈. 병원을 대학과 분리하여 집계

    hospital = True 독립. 대학병원을 대학에서 독립해서 다룸

    """

    # 1. name 에 병원 단어(병원, 의료원, 의원) 포함 하는가 ? 병원, 의료원 이 포함 없으면 종료함
    if re.search(hospital_names_ptn, name):
        pass
    else:
        #print(f"1.........name = {name}")
        return name

    # 2. 대학교와 대학병원을 분리할 것인가 ?
    if hospital == False:
        # 병원을 독립 시키지 않을 때, 즉 대학병원에 속한 병원은 대학명을 리턴 시킬 때,
        if hospital_df is None :
            # 사전 없을 때
            name = name.replace(' ', '')
            if name in  ['분당서울대학병원','분당서울대학병원','분당서울대병원']:
                name = '서울대학'
            elif name in ['가천대학교의료원']:
                name = '가천대학'
            return name
        else: # 사전 가지고 있을 때, 병원에 해당하는 대학 리턴하기
            name2 = find_hospital_name(name=name,
                                       df_dic=hospital_df,
                                       name_lemma='병원이름',
                                       name_univ='대학명',
                                       name_others='별칭',
                                       show=False,
                                       kind='u')
            return name2

    """ 병원명 정제 """
    #3 사전 기반
    if hospital_df is None:
        # 사전 없으면 패턴 기반으로 진행함
        if ptn_list is None:
            ptn_list = global_ptn_hospital_list
        name2 = None
        # if ptn_list == []:
        #    # 기본 패턴 적용
        #    ptn_list = ['^([\w\s]+병원|의료원|한의원|의원)']
        for i, each_ptn in enumerate(ptn_list):
            temp = re.findall(each_ptn, name.strip())
            if temp:
                if show_ptn == True:
                    print(f"{i} : {each_ptn}")
                if isinstance(temp[0], tuple):
                    name2 = temp[0][0]
                else:
                    name2 = temp[0]
                break
        if name2 is not None:
            name = name2

    else:
        name = find_hospital_name(name=name,
                               df_dic=hospital_df,
                               name_lemma='병원이름',
                               name_univ='대학명',
                               name_others='별칭',
                               show=False,
                               kind='h')


    #print(f"2.........name = {name}")
    if re.search(r'대학교\s?병원$', name):
        x1 = re.sub('대학교\s?병원$', '대학병원', name)
    elif re.search(r'대학\s?병원$', name):
        x1 = re.sub('대학\s?병원$', '대학병원', name)
    elif re.search('대\s?병원$', name):
        x1 = re.sub('대\s?병원$', '대학병원', name)
    elif re.search(r'대학교?\s?치과\s?병원$', name):
        x1 = re.sub('대학교?\s?치과\s?병원$', '대학치과병원', name)
    elif re.search(r'대학교\s?의료원$', name):
        x1 = re.sub('대학교\s?의료원$','대학의료원',name)
    else:
        x1 = name
    #print(f'before : {name} after : {x1}')
    return x1



#@deco1
def replace_university_name(name=None, show_ptn=False, hospital=False):
    """대학 이름 부분을 특화 시켜 전처리 해줌. 2022.12.14(WED)"""

    if hospital == True:
        # 병원, 의료원 이 포함되면 종료함
        if re.search(r"병원|의료원|한의원", name) :
            #print(f"{name} 은 병원이기에 대학명 분석에서는 제외합니다")
            return name

    # 대학, 대학원 대상으로 함
    if re.search('대|대학|대학교|대학원|과학기술원|사관학교', name):
        pass
    else:
        #print(f"{name} 은 대학이 아니기에 대학명 분석에서는 제외합니다")
        return name

    name_raw = name
    # 대학원대학 문제
    ptn1 = r"([가-힣\w]+대학원대학교?)$"
    ptn2 = r"([가-힣\w]+대학원대학교)\s?([가-힣\w]+)$"

    # 협의회 문제
    ptn2a = r"([\w\s·]+대학[\w\s·]+[협의회|협회])$"
    # '서울불교대학원대학교 불교와심리연구원',

    # 산학협력단 문제 : 국민대학교 산학협력단, 서울과학기술대학원 산학협력단, 연세의료원산학협력단
    ptn3 = r"([가-힣\w]+대학교?)\s?(\(?[가-힣\w]+\s?캠퍼스)\)?\s?([가-힣\w]+협력단)$" # 00대학교 00캠퍼스 산학협력단
    #ptn4 = r"([가-힣\w]+(의료원|대학원|대학교|교대|대학|대))\s?((글로컬산학|연구산학|산학|산학|산합|산업)협력단)$" # 00대학교 산학협력단, 한국외국어대학교 연구산학협력단, 서울교대 산학협력단
    ptn4 = r"([가-힣\w]+(대학원|대학교|교대|대학|대))\s?((글로컬산학|연구산학|산학|산학|산합|산업)협력단)$"
    #ptn5 = r"^([\w\s]+대학교)\s?((글로컬|의료원|에리카|\w+)\s?(연구산학|산학|산합)협력단)"
    ptn5 = r"^([\w\s]+대학교)\s?((글로컬|에리카|\w+)\s?(연구산학|산학|산합)협력단)"  # 건국대학교 글로컬산학협력단, 제주대학교산합협력단
    ptn6 = r"^([\w\s]+[대학교|대학])\s?(산학\s?협력단)([\w\s_]*)"

    # 캠퍼스 문제
    ptn7 = r"([가-힣\wI]+대학교?)\s?\(?([가-힣\w]+캠퍼스\)?)$" # 성균관대학교(자연과학캠퍼스), 한국폴리텍I대학서울강서캠퍼스

    # 대학교 단과대학 문제
    ptn8 = r"([가-힣\w]+대학교?)\s?([가-힣\w]+대학)$"
    ptn9 = r"([가-힣\w]+대학교?)\s?([가-힣\w]+대학)\s?([\w\s]+)(?!병원|의료원|한의원)$"

    # 000대학교00연구소, 00대학교 00대학 00연구소, 00대학교 Human-inspired 복합지능 연구센터, '경인교육대학교학습및정서·행동문제연구소'
    ptn13 = r"([가-힣\w\s]+대학교?)\s?([\w\s\-·]+)$"
    ptn13 = r"([가-힣\w\s]+대학교?)\s?([\w\s\-·]+)(?!병원|의료원|한의원)"
    # (HW) : 부산대학교 부산지리연구소 (미사용)
    ptn14 = r"([가-힣\w]+대학교?)\s?([\w\s\-\(\)]+)$"
    #print('ptn9a=',ptn9a)

    # 병원 문제
    #ptn10 = r"([가-힣\w]+대학교?)\s?([가-힣\w]+대학)\s?[가-힣\w]+병원$" # 고려대학의과대학부속병원
    #ptn11 = r"([가-힣\w]+대학교?)\s?([가-힣\w]+병원)$"
    #ptn12 = r"([가-힣\w]+대학교?)\s?([가-힣\w]+병원)\([가-힣\w]+\)$"
    #ptn12a = r"([\w\s]+병원)$"

    # 과학기술원 : 한국, 대구경북, 울산, 광주
    ptn15 = r"^([\w\s]+과학기술원)(?!\s?부설|고등과학원|선박해양플랜트연구소)"

    # 한국연구원 추출
    #ptn16 = r"^(한국[\w]+연구원)\s?(?!부설)" # 부설은 제외
    ptn16 = r"(한국[\w]+연구원)(?!\s?부설)"
    ptn_ = r"(^[가-힣\wI\s]+대학교?)$" # 00대학교대

    ptns = [ ptn1, ptn2, ptn2a,
             ptn3, ptn4, ptn5,
             ptn6, ptn7, ptn8, ptn9,
             #ptn10,ptn11, ptn12, # 병원
             #ptn12a,
             ptn13, ptn14, ptn15, ptn16]
    x_trans = None
    for i, each_ptn in enumerate(ptns):
        temp = re.findall(each_ptn, name.strip())
        if temp:
            if show_ptn == True:
                print(f"{i} : {each_ptn}")
            if each_ptn in [ptn1, ptn2a, ptn15, ptn16]:
                if isinstance(temp[0], tuple):
                    x2 = temp[0][0]
                else:
                    x2 = temp[0]
                #print(f"x2 = {x2}, type({type(x2)}")
            #elif each_ptn in [ptn12a]: # 병원
            #    x2 = temp[0]
                
            else:
                x2 = temp[0][0]
                # 00대 산학협력단 문제 해결, 00대 로 리턴되는 것을 00대학으로 리턴하게 함
                if each_ptn in [ptn3, ptn4, ptn5, ptn6] :
                    if x2.endswith('교대'):
                        pass
                    elif x2.endswith('여대'): # 이화여대 산학협력단
                        pass
                    elif x2.endswith('대'): # 서울대 산학협력단 -> 서울대학
                        x2 = x2 + '학'
            x_trans = x2
            break
    #return x3
    # 마지막
    if x_trans is None:
        temp = re.findall(ptn_, name)
        if temp:
            x_trans = temp[0]
        else:
            x_trans = name

    # print('x_trans = ', x_trans)
    # 마지막 단어 대학교를 대학으로 변경 하기
    x4 = re.sub('대학교$', '대학', x_trans.strip())

    # 마지막 단어 여자대학을 여대로 변경하기
    x4a = re.sub('여자대학$', '여대', x4.strip())

    # 마지막 단어 교육대학을 교대로 변경하기
    x5 = re.sub('교육대학$', '교대', x4a.strip())

    # 맨 앞 국립 있으면 제거하기 : 국립서울대학 --> 서울대학
    x6 = re.sub('^국립', '', x5.strip())

    return x6.strip()

@deco1
def replace_company_name2(name=None, ptn_list=None, show_ptn=False):
    """ 기업명 정제 """
    name2 = None
    #print('ptn_list = ', ptn_list)
    if ptn_list is None:
        # 기본 패턴 적용
        #print('ptn_list is None')
        ptn_list = global_ptn_company_list
        #return name
        #ptn_list = ['^([^농업][\w\s]+)\(주\)[\w\s]+']

    for i, each_ptn in enumerate(ptn_list):
        #print(f"i={i}")
        temp = re.findall(each_ptn, name.strip())
        if temp:
            if show_ptn == True:
                print(f"{i} : {each_ptn}")
            if isinstance(temp[0], tuple):
                name2 = temp[0][0]
            else:
                name2 = temp[0]
            break
    if name2 is not None:
        return name2
    else:
        return name



@deco1
def replace_university_name2(name=None, ptn_list=None, show_ptn=False, hospital=False,
                             univ_ptn ='대|대학|대학교|대학원|과학기술원|사관학교|학교',
                             hospital_ptn="병원|의료원|한의원"):
    """대학 이름 부분을 특화 시켜 전처리 해줌. 2022.12.14(WED)
    2023.2.4 사용자 패턴을 입력받게 해줌
    """

    if hospital == True:
        # 병원, 의료원 이 포함되면 종료함

        if re.search(hospital_ptn, name):
            #print(f"{name} 은 병원이기에 대학명 분석에서는 제외합니다")
            return name
    else:
        # 대학으로 집계할 때, 분당서울대학교 병원을 서울대학으로 명칭변경하여 집계함
        if name == '분당서울대학교병원' :
            return '서울대학'

    # 대학, 대학원 대상으로 함
    if re.search(univ_ptn, name):
        pass
    else:
        #print(f"{name} 은 대학이 아니기에 대학명 분석에서는 제외합니다")
        return name

    # ptn_list 가 None 이면, 디폴트 패턴 리스트 불러오게 함
    if ptn_list is None :
        ptn_list = global_ptn_univ_list

    ptn_ = r"(^[가-힣\wIV\s]+대학교?)$"  # 00대학교

    x_trans = None
    for i, each_ptn in enumerate(ptn_list):
        temp = re.findall(each_ptn, name.strip())
        if temp:
            if show_ptn == True:
                print(f"{i} : {each_ptn}")
            if isinstance(temp[0], tuple):
                x2 = temp[0][0]
            else:
                x2 = temp[0]
            if re.search('(산학|산합)협력단', name):
                # 00대 산학협력단 문제 해결, 00대 로 리턴되는 것을 00대학으로 리턴하게 함
                if x2.endswith('교대'):
                    pass
                elif x2.endswith('여대'):  # 이화여대 산학협력단
                    pass
                elif x2.endswith('대'):  # 서울대 산학협력단 -> 서울대학
                    x2 = x2 + '학'
            x_trans = x2
            break
    # 마지막
    if x_trans is None:
        temp = re.findall(ptn_, name)
        if temp:
            x_trans = temp[0]
        else:
            x_trans = name

    # print('x_trans = ', x_trans)
    # 마지막 단어 대학교를 대학으로 변경 하기
    x4 = re.sub('대학교$', '대학', x_trans.strip())

    # 마지막 단어 여자대학을 여대로 변경하기
    x4a = re.sub('여자대학$', '여대', x4.strip())

    # 마지막 단어 교육대학을 교대로 변경하기
    x5 = re.sub('교육대학$', '교대', x4a.strip())

    # 맨 앞 국립 있으면 제거하기 : 국립서울대학 --> 서울대학
    x6 = re.sub('^국립', '', x5.strip())

    return x6.strip()


def find_ptn1(ptn='', txt=''):
    """패턴에 해당하는 것이 있으면 그 것을 리턴해 줌"""
    try:
        if ptn == '':
            return txt
        temp = re.findall(ptn, txt)
        if len(temp)>0:
            print(f"temp={temp}")
            return temp[0] # 처음것만 리턴
        else:
            return None
    except Exception as err:
        print(f"error = {err}")

def replace_with_user_ptn(ptnlist=[], txt='', option='no'):
    """패턴에 해당하는 것이 있으면 그 것을 리턴해 줌
    option : 디폴트는 'no', 패턴 번호를 리턴함. 패턴에 해당되는 것이 없으면 -1을 리턴함
    """
    for i, ptn in enumerate(ptnlist):
        if ptn == '':
            if option == 'no':
                return -1
            else:
                return txt

        temp = re.findall(ptn, txt)
        if len(temp)>0:
            print(f"temp={temp}")
            if option == 'no':
                return i  # temp[0][0]
            elif option == 0 :
                return temp[0] # 처음것만 리턴
            else:
                return temp[0][0]

    if option == 'no':
        return -1
    else:
        return txt

#def replace_org_name_with_user_dic(x, dic=None):
def replace_name_with_user_dic(x, dic=None, kind=1):
    """
    사용자 사전에 기반하여 기관명 변경 
    키에 해당하는 기관명을 입력받아서, 수정한 기관명으로 리턴해 줌
    키에 해당되지 않으면, 입력한 값을 그대로 리턴해 줌

    kind : 1 이면 리턴할 때 사전에 없는 단어는 그냥 그래로 리턴함
           2 이면 리턴할 때 사전에 없는 단어는 '' 리턴함
    
    2022.12.14(WED)
    2022.12.16(FRI)
    """
    if dic is None:
        # 기관명 사전 
        dic = dic_org
        
    if x in dic.keys():
        return dic[x]
    else:
        if kind == 1:
            return x
        elif kind == 2:
            return ''
        else:
            return x

@deco1
def remove_prefix_suffix(prefix_list=[], suffix_list=[], name='', n=1):
    """ 접두어, 접미어 처리 함수(prefix, suffix remove) 2023.2.3
        n : 농업회사법인 (주) 기업명 경우 처리하기 위하여 만듦
    """
    #print(f"* len(prefix_list)={len(prefix_list)}, len(suffix_list={len(suffix_list)}")
    for i in range(n):
        for each in prefix_list:
            if name.startswith(each):
                name = name[len(each):].strip()
        for each in suffix_list:
            if name.endswith(each):
                name = name[:-1 * len(each)].strip()
    return name


def clean_kw(x, sep=','):
    """키워드 정제용 함수 """
    if x is None:
        return ''
    if isinstance(x, float):
        return ''
    else:
        ptn1 = r"[\n\t]+"
        x2 = re.sub(ptn1, sep, x)  # (ex) 교류직결형,\t균일도,표준모듈,플리커,빛공해 -> 교류직결형,,균일도,표준모듈,플리커,빛공해
        x3 = x2.split(sep)
        x4 = [each for each in x3 if each != '']
        x5 = f'{sep}'.join(x4) # remove double comma(,) 
        return x5

def make_table_no1(df,
                   columns1=['DP(정제)','TI(정제)','사업명(정제)','내역사업명(정제)'], 
                   col_PY ='PY', 
                   col1='FUND', 
                   col_id='과제고유번호',
                   col_kw='KW1(정제)',
                   col_group='중분류(싱글)',
                   type_no=1):
    
    """과제개수, 과제금액, 과제고유번호 테이블을 생성하여 리턴함
    
    2022.12.8(THR)
    """
    #print(f"df.columns = {df.columns}")
    print(">>> make_table_no1()")
    df_list = []
    
    columns = columns1 + [col_PY]
    dfa = df.groupby(columns)[col1].agg('sum').unstack(fill_value=0)/1e8
    
    dfa['합계(억원)'] = dfa.sum(axis=1)
    for each in dfa.columns:
        dfa[f'rank_{each}'] = dfa[each].rank(axis=0, method='min',ascending=False)
    df_list.append(dfa)

    dfb = df.groupby(columns)[col1].agg('count').unstack(fill_value=0)
    dfb.columns = [f"{each}a" for each in dfb.columns] # 컬럼명 중복되면 소팅하는 데 어려움 발생함 
    dfb['합계(개)'] = dfb.sum(axis=1)
    df_list.append(dfb)
    
    df[col_id]=df[col_id].astype(str)
    dfc = df.groupby(columns1)[col_id].apply(lambda x : ';'.join(x))
    dfc2 = DataFrame(dfc)
    dfc2['과제수'] = dfc2[col_id].map(lambda x : len(x.split(';')))
    df_list.append(dfc2)
   
    def kw_merge(x):
            """ 
            2022.12.17(SAT)
            """
            if x is None:
                return ''
            elif isinstance(x, float):
                return ''
            else:
                temp=[]
                for each in x:
                    x3 = each.split(',')
                    x4 = [each.strip() for each in x3 if each.strip() !='']
                    temp.extend(x4)
                txt = ';'.join(sorted(set(temp)))
                return txt 
            
    if col_kw in df.columns: 
        df[col_kw]=df[col_kw].astype(str)    
        dfd = df.groupby(columns1)[col_kw].apply(lambda x : kw_merge(x))
        dfd2 = DataFrame(dfd)
        #dfd2['과제수'] = dfc2[col_id].map(lambda x : len(x.split(';')))    
        df_list.append(dfd2)
 
    if col_group in df.columns:
        df[col_group]=df[col_group].astype(str)    
        dfe = df.groupby(columns1)[col_group].apply(lambda x : kw_merge(x))
        dfe2 = DataFrame(dfe)
        #dfd2['과제수'] = dfc2[col_id].map(lambda x : len(x.split(';')))    
        df_list.append(dfe2)
        


    df_r = pd.concat(df_list, axis=1)
        
    # 내보내는 형식 지정 : 컬럼 개수 2개로 제한함
    if type_no == 2 :
        df_r2 = df_r[['합계(억원)', '합계(개)']].copy()  
        df_r = df_r2
    #else:
    #    print(f" type_no = {type_no}")
    #    pass 
        
    return df_r


# 파일 이름 지정
#def make_format1(df:DataFrame, ptn_kw=None):
def to_excel_form1(df:DataFrame, ptn_kw=None, df_r=None, subdir=None, unit=8, filename=None, sheet_include_data=True):
    """ df_output2 입력 받아서 엑셀파일로 자동 저장해 주기 
    unit : fund 단위 : 디폴트 8, 억원 기준
    sheet_include_data : data sheet 포함할 지 여부를 지정함. 디폹트는 True
    """
    print(">>> make_format2() from bk_ntis_find.py")
    
    df = df.copy() 
    # file name 
    a = datetime.datetime.now()
    #a2 = a.strftime('%Y_%m%d_%Hh%Mm_%S') 파일이름이 너무 길어서 년도정보는 생략 
    a2 = a.strftime('%m%d_%Hh%Mm_%S')
    if ptn_kw is None:
        ptn_kw = '(sample)'
    if filename is not None:
        fname_xlsx = filename
    else:
        if subdir is not None :
            fname_xlsx = rf"{subdir}/df_stage1_{ptn_kw}_({df.shape[0]}_{df.shape[1]})_{a2}.xlsx"
        else:
            fname_xlsx = rf"./df_stage1_{ptn_kw}_({df.shape[0]}_{df.shape[1]})_{a2}.xlsx"


    # xlsxwriter는 제로 인덱스 사용함 : 0 부터 시작함
    # 엑셀은 1 부터 시작 
    writer = pd.ExcelWriter(fname_xlsx, engine='xlsxwriter')
    #---------------------------------------
    # data  preparation
    # --------------------------------------
    dic_result = {}
    
    col_ID = '과제고유번호'
    col_dp = 'DP'
    col_dp2 = 'DP(정제)'
    col_ti = 'TI(정제)'
    col_py = 'PY'
    col_p1 = '사업명(정제)'
    col_p2 = '내역사업명(정제)'
    col_fund = 'FUND'
    col_fund2 = col_fund + '(억원)'
    col_org1 = '과제수행기관명'
    col_org = '과제수행기관명(정제)' 
    col_count = '과제수'
    col_name = 'name'
    col_name_float = "합계(억원)"
    col_name_int = "합계(개)"
    col_kw1 = 'KW1(정제)'
    col_kw2 = 'KW2(정제)'
    col_AR = 'AR(정제)'
    
    col_name_ratio = 'ratio'
    col_name_float2 = "합계(억원)_total"
    col_name_float3 = "합계(억원)_sub"
    col_name_pr = "(부처)사업명"
    
    col_group = '중분류'
    col_group_single='중분류(싱글)'
    col_group_multi='중분류(멀티)'
    
    default_dic_col_width = {
        col_ID:15,
        col_dp:15,
        col_dp2:8,  
        col_p1:30,
        col_p2:40,
        col_ti:50,
        col_fund:10,
        col_fund2:10,
        col_py:10,
        col_org1:20,
        col_org:20,
        col_count:5,
        col_name:30,
        col_kw1:30,
        col_kw2:30,
        col_name_ratio:10,
        col_name_float2:10,
        col_name_float3:10,
        col_name_pr:30,
        col_group:30,
        col_group_single:5,
        col_group_multi:10
    }
    
    w1 = 6 # 폭 
    col_number1 = df[col_py].min()
    col_number2 = df[col_py].max()
    print(f"* {col_py} : col_number1 = {col_number1}, col_number2 = {col_number2}")
    
    print(f"* col_fund = {col_fund}")
    if col_fund in df.columns:
        if unit ==8:
            print(f"* 펀딩 단위 : 억원, {col_fund}")
            df[col_fund2]= df[col_fund]/1e8
        
    # sheet name : data 
    # data sheet 를 넣으면 데이터 크기가 커질 수 있음. 선택하게 해줌
    if sheet_include_data == True:
        writer = be.make_excel_from_writer_bar(writer=writer,
                                           df=df, 
                                           sheet_name="data", 
                                           index_name='번호',
                                           col_name_int=col_py,
                                           dic_column_width= default_dic_col_width,  
                                           kind=0)
    # sheet name : (집중사업)                                      
    if df_r is not None:
        print(f"* df_r.shape = {df_r.shape}")
        df_r2 = df_r.reset_index()
        df_r2 = df_r2.rename(columns={'index':col_name_pr})
                                  
        writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_r2, 
                                           col_name_float=col_name_float3, col_name_int=col_name_ratio, 
                                           sheet_name="(집중)",  
                                           col_name_categories=col_name_pr, pos_x = 3, 
                                           figsize=(1200,300),
                                           # index_name='부처명',
                                           axis_name_left='투자액(억원)', axis_name_right='비율(%)',  
                                           kind='sheet+column',
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2021:w1,2022:w1},
                                           #column_number=(col_number1, col_number2),
                                           name_font_size=8,
                                           num_font_rotation=90,
                                          )

    # sheet name : Table1 
    df_table1 = make_table_no1(df, columns1=[col_py], type_no=2)
    #df_table1['name'] = df_table1.index 
    df1=df_table1.reset_index()
    dic_result.update({'table1':df1})
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df1, 
                                           col_name_float=col_name_float, 
                                           col_name_int=col_name_int, 
                                           sheet_name="1(년도)", 
                                           col_name_categories=1, pos_x = 3, name_font_size=8, 
                                           figsize=(800,300),
                                           axis_name_left='투자액(억원)', axis_name_right='과제수',
                                           index_name='년도',
                                           dic_column_width=default_dic_col_width,
                                           column_number=(col_number1, col_number2),
                                           kind='sheet+column')
        
    # sheet name : Table 2 : 부처, 년도별 투자금액
    df_table2a = make_table_no1(df, columns1=[col_dp, col_dp2], type_no=1)
    df_table2 = df_table2a.reset_index()
    dic_result.update({'table2':df_table2})
    #df2 = df_table2a.sort_values(by=[col_name_float], ascending=[False]) 
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_table2, col_name_float=col_name_float, col_name_int=col_name_int, 
                                           sheet_name="2(부처)",  
                                           col_name_categories=col_dp2, pos_x = 3, name_font_size=8, 
                                           figsize=(1200,300),
                                           # index_name='부처명',
                                           axis_name_left='투자액(억원)', axis_name_right='과제수',  
                                           kind='sheet+column',
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                           column_number=(col_number1, col_number2),
                                          )
    
    # sheet name : Table 3 : 부처, 사업별 년도별 투자금액
    df_table3a = make_table_no1(df, columns1=[col_dp2, col_p1])
    df_table3 = df_table3a.reset_index()
    dic_result.update({'table3':df_table3})
    #df_table3['name'] = df_table3[col_p1] + '(억원)' + df_table3[col_dp2] +')'
    #df_table3['name'] = df_table3[col_p1] + df_table3[col_dp2]
    #df_table3 = df_table3a.sort_values(by=[col_name_float], ascending=[False])
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_table3, col_name_float=col_name_float, col_name_int=col_name_int, 
                                           col_name_categories=1,
                                           sheet_name="3(사업)",  
                                           pos_x = 3, name_font_size=8, max_n=100,
                                           kind=0,
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                           column_number=(col_number1, col_number2),
                                           column_number_cr=5,
                                           )
    
    # sheet name : Table 4 : 부처, 사업별, 내역사업별 년도별 투자금액
    df_table4a = make_table_no1(df, columns1=[col_dp2, col_p1, col_p2])
    df_table4 = df_table4a.reset_index()
    dic_result.update({'table4':df_table4})
    #df_table4['name'] = df_table4[col_p1] + '(' + df_table4[col_p2] +')'
    #df_table4['name'] = df_table4[col_p1] + '(' + df_table4['내역사업명(정제)'] +')'
    #df_table3 = df_table3a.sort_values(by=[col_name_float], ascending=[False])
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_table4, col_name_float=col_name_float, col_name_int=col_name_int, 
                                           col_name_categories=1,
                                           sheet_name="4(내역사업)",  
                                           pos_x = 3, name_font_size=8, max_n=30,
                                           kind=0,
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                           column_number=(col_number1, col_number2),
                                           column_number_cr=5,
                                           
                                           ) 
    
    # sheet name : Table 5 : 부처, 사업별, 내역사업별, 과제제목 년도별 투자금액
    df_table5a = make_table_no1(df, columns1=[col_dp2, col_p1, col_p2, col_ti, col_org])
    df_table5 = df_table5a.reset_index()
    dic_result.update({'table5':df_table5})
    #df_table5['name'] = df_table5[col_ti] + '(' + df_table5[col_org] +')'
    #df_table3 = df_table3a.sort_values(by=[col_name_float], ascending=[False])
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_table5, col_name_float=col_name_float, col_name_int=col_name_int, 
                                           col_name_categories=1,
                                           sheet_name="5(과제)",  
                                           pos_x = 3, name_font_size=8, max_n=30,
                                           kind=0,
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                           column_number=(col_number1, col_number2),
                                           column_number_cr=1,
                                           )  
    
    # sheet name : Table 6 : 연구과제수행기관(정제)별 년도별 투자금액
    df_table6a = make_table_no1(df, columns1=[col_org])
    df_table6 = df_table6a.reset_index()
    df_table6.sort_values(by=['합계(억원)'], ascending=[False], inplace=True)
    dic_result.update({'table6':df_table6})
    #df_table6['name'] = df_table6.index 
    #df_table3 = df_table3a.sort_values(by=[col_name_float], ascending=[False])
    writer = be.make_excel_from_writer_bar(writer=writer,  
                                           df=df_table6, 
                                           col_name_float=col_name_float, 
                                           col_name_int=col_name_int, 
                                           col_name_categories=1,
                                           sheet_name="6(수행기관)", 
                                           pos_x = 3, name_font_size=8, max_n=15,
                                           kind='sheet+bar+column',
                                           axis_name_left='연구과제수행기관',
                                           axis_name_right='과제수',
                                           dic_column_width=default_dic_col_width,
                                           dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                           column_number=(col_number1, col_number2),
                                           column_number_cr=1, 
                                           )  



    # sheet name : Table 7 : 인덱스와 컬럼명이 중복되어 에러 발생할 수 있음. 주의 필요.
    if col_group_single in df.columns:
        df_table7a = make_table_no1(df, columns1=[col_group_single]) 
        df_table7a = df_table7a.rename(columns={col_group_single: col_group_single+'2'})
        df_table7 = df_table7a.reset_index()
        #df_table.sort_values(by=['합계(억원)'], ascending=[False], inplace=True)
        dic_result.update({'table7':df_table7})
        #df_table6['name'] = df_table6.index 
        #df_table3 = df_table3a.sort_values(by=[col_name_float], ascending=[False])
        writer = be.make_excel_from_writer_bar(writer=writer,  
                                            df=df_table7, 
                                            col_name_float=col_name_float, 
                                            col_name_int=col_name_int, 
                                            col_name_categories=1,
                                            sheet_name="7(중분류)", 
                                            pos_x = 3, name_font_size=8, max_n=15,
                                            kind='sheet+bar+column',
                                            axis_name_left='중분류',
                                            axis_name_right='과제수',
                                            dic_column_width=default_dic_col_width,
                                            dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                            column_number=(col_number1, col_number2),
                                            column_number_cr=1, 
                                            )  

    # sheet name : Table 8 : 
    if col_AR in df.columns:
        df_table8a = make_table_no1(df, columns1=[col_AR]) 
        df_table8  = df_table8a.reset_index()
        dic_result.update({'table8a':df_table8})
        writer = be.make_excel_from_writer_bar(writer=writer,  
                                            df=df_table8, 
                                            col_name_float=col_name_float, 
                                            col_name_int=col_name_int, 
                                            col_name_categories=1,
                                            sheet_name="8(AR)", 
                                            pos_x = 3, name_font_size=8, max_n=15,
                                            kind='sheet+bar+column',
                                            axis_name_left='지역',
                                            axis_name_right='과제수',
                                            dic_column_width=default_dic_col_width,
                                            dic_column_number={2017:w1,2018:w1,2019:w1,2020:w1,2021:w1,2022:w1},
                                            column_number=(col_number1, col_number2),
                                            column_number_cr=1, 
                                            x_offset=5, y_offset=10,
                                            ) 

    # excel file save 
    writer.save()
    print(f"{fname_xlsx} was successfully saved")
    
    return dic_result
    
    
    


def make_result_stage1(df,
                        PY = (2018,2022), 
                        ptn_kw = None,  
                        search_class = [ '(1_분류)', '(2_사업명내역사업명', '(3_키워드)', '(4_TI)'],
                        search_columns = [ 'code_all',['사업명(정제)','내역사업명(정제)'], ['KW1'], ['TI(정제)']],
                        col1 ='DP',
                        col2 = '사업명(정제)',
                        col_PY ='PY',
                        col_fund = 'FUND',
                        col_TI = 'TI(정제)',
                        
                        show=True,
                        stopword_include=False,
                        stopword_TI = '보안과제정보',
                        col_stop_fields = ['KW1(정제)','KW2(정제)','TI(정제)'],
                        ptn_stop = None,
                        by='any'
                      ):
    """ 
    stage1 결과를 저장해 주는 함수
    """
    print(">>> make_result_stage1()")
    t1 = time.time()
    
    if ptn_kw is None:
        print("* check your data : ptn_kw is None !")
        return False 
    
    # 조건 :  년도
    df = df[(df[col_PY] >= PY[0]) & (df[col_PY] <= PY[1])].copy() 
    
    result_list = []
    printa(f"# 패턴 : {ptn_kw}, 전체 과제수 : {df.shape} ", symbol='*', show=show, kind=2) 
    printa(f"PY={PY}", symbol='*', show=show, kind=1)

    n_security = (df[col_TI] == stopword_TI).sum()
    if stopword_include : # 
        printa("# 0a. '보안과제정보' 포함함 ", show=show)    
        if show :
            print(f"{stopword_TI}({n_security:,}개)는 과제명에 일괄적으로 '{stopword_TI}'로 명시되어 있음")
    else: 
        printa(f"# 0a. 과제이름에서 '보안과제정보'({n_security:,}개) 제거함", show=show)
        dfb = df[df[col_TI] != stopword_TI].copy()
        df = dfb
        print(f"*** df.shape = {df.shape}")
        
        if ptn_stop is not None: 
            printa(f"# 0b. 키워드({col_stop_fields} 에서 {ptn_stop} 제거함", show=show)
            onoff_list_kw =[]
            for each_col_kw in col_stop_fields:
                onoff = df[each_col_kw].str.contains(ptn_stop, na=False, case=False)
                onoff_list_kw.append(onoff)
            if len(onoff_list_kw)>0:
                print(f"*** before : {df.shape}")
                n1 = df.shape[0]
                onoff1 = onoff_list_kw[0]
                for i in range(1, len(onoff_list_kw)):
                    print(f"{i} = {onoff1.sum()}")
                    onoff1 = onoff1 | onoff_list_kw[i]
                    
                df2 = df[~onoff1]
                df = df2.copy() 
                n2 = df.shape[0]
                print(f"*** after : {df.shape}")
                print(f"증감 : ab {n1:,} ad {n2:,}, {n1-n2:,}개 감소")
        

        
    printa("1. 반복하여 지정한 컬럼에서 패턴을 찾아내는 작업 시작함", show=show)
    for prefix, each_colnames in zip( search_class, search_columns ):
        printa(f"{prefix} : {each_colnames}", symbol='.', kind=1)

        result = find_info(df=df, ptn=ptn_kw, column_names=each_colnames, col_fund=col_fund,
                            col1=col1, col2=col2, col_PY=col_PY,
                            save=True, file_prefix=ptn_kw + prefix)
        result_list.append(result['con'])


    printa("# 2. 조건 통합한 결과 ", show=show)
    df_all = pd.concat(result_list, axis=1)
    if by == 'any':
        con_all = df_all.any(axis=1) 
    else:
        con_all = df_all.all(axis=1) 
        
    n_con_all = con_all.sum()
    df['TF1'] = con_all  # TF 필드 생성
    if show:
        print(f"* 과제수(조건 ({len(search_class)}가지 부문 만족하는 개수) : {df['TF1'].sum() :,}개")

    printa(f"# 3. 과제 제목이 같은 과제 추가해주기 : 기준은 '부처명_과제명'({col1, col_TI}) 기준으로 함", show=show)
    # 과제제목 일치할 때 기준이 되는 컬럼 생성 : dp_ti = DP + TI(정제)
    # df['dp_ti'] = df['DP'] + df['TI(정제)']
    df['dp_ti'] = df[col1] + df[col_TI]
    set_ti_all = set()
    for i, each_result in enumerate(result_list) :
        df_temp = df[each_result].copy() # 워닝 제거하기 위해서 copy() 해줌. 검색된 과제를 추출함
        #df_temp['dp_ti'] = df_temp['DP'] + df_temp['TI(정제)'] 
        ti_list1 = df_temp['dp_ti'].to_list() 
        # 과제 모두 합하기
        set_ti_all |= set(ti_list1)
        if show:
            print(f" - {i+1}차 {search_class[i]} : 과제수 {len(ti_list1):,}개, 유니크 과제건수 {len(set(ti_list1)):,}개")
            
    df['TF2'] = df['dp_ti'].isin(set_ti_all) # TF 필드 생성 : 과제제목 리스트에 포함되어 있는가 ? 
    df_stage1 = df[df['TF2']].copy() # 제목 집합에 포함된 과제만 추출
    if show :    
        print(f"* 과제수(같은 제목 추가한 것 반영) : {df['TF2'].sum():,}개")
        print(f"※ 제목 비교 과정을 통해서 {df['TF2'].sum() - df['TF1'].sum() :,}개 증가함")
    

    
    printa("4. 결과", show=show)
    print(f"* 과제수(df_stage1) : {df_stage1.shape[0]:,}개, 컬럼수 : {df_stage1.shape[1]:,}개")
    print(f"* consumed : {time.time() - t1 :.1f} seconds")
    #df_stage1[['과제고유번호']].to_csv(f"{ptn_kw}{df_stage1.shape}.csv")

    return df_stage1

# 파일 저장
# 출력할 컬럼 지정 

def make_selected_columns(df=None, selected_columns = None,  
                          by_columns=['DP','TI(정제)','PY'], 
                          kind=None,
                          plus = []):
    """
    선택한 컬럼(selected_columns)을 추출하고, by_columns에 따라 정렬함
    """
    print(">>> make_selected_columns( ... )")
    print('1 추출 전 :', df.shape) 
    
    match kind:
        case 1|'all'|'A':
            selected_columns = columns_total + plus 
            print(f"selected_columns = {selected_columns}")
        case 2|'basic'|'B':
            selected_columns = columns_basic + plus 
            print(f"selected_columns = {selected_columns}")        
        case _ : 
            if selected_columns is not None:
                selected_columns = selected_columns + plus 
            else:
                print("check your data in parameters")
                return df 
    # 선택 컬럼이 df 의 컬럼에 있는 것만 다시 선택하기 
    selected_columns2 = [ each for each in selected_columns if each in df.columns]
    
    # 정렬 기준 컬럼이 df 의 컬럼에 있는 것만 다시 선택하기 
    by_columns2 = [ each for each in by_columns if each in df.columns]
    ascending = [ True for each in by_columns2]
    
    if (len(selected_columns2)>0) and (len(by_columns2)>0):
        df_output = df[selected_columns2].sort_values(by=by_columns2,ascending=ascending).copy()
        df_output2 = df_output.fillna('-')
        df_output2.index = pd.RangeIndex(len(df_output2.index))
        print('2 추출 후:', df_output2.shape)
        return df_output2 
    else:
        return df 


def make_stage1_to_excel(df_stage1, kind=2, df_r=None, filename=None):
    """ df_stage1 받아서 컬럼선택, 컬럼정제, 엑셀파일저장 진행하는 함수
    Args:
        df_stage1 (_type_): 
        kind (int, optional): 'all', 'basic' : 핵심적인 컬럼만 추출함
        df_r : 새로운 정보 
        filename (_type_, optional):
    Returns:
        DataFrame: 데이타프레임 결과를 리턴해 줌
    """
    print(">>> make_stage1_to_excel(...)")
    # 출력할 컬럼 선택 하기 : df_output
    dic_result = {}
    df_output = make_selected_columns(df_stage1, kind=kind)
    dic_result.update({'data':df_output})

    for each in ['AB','AB2','EF']:
        if each in df_output.columns:
            df_output = df_output.drop(columns=[each])
    print(f"* 데이터 shape : {df_output.shape}")
    #df_output.columns
    
    if filename is not None:
        dic1 = to_excel_form1(df_output, df_r=df_r, ptn_kw=filename) 
        dic_result.update(dic1)
    return dic_result 
    
def make_reference_dic(df, 
                       col_target='사업명(정제)', 
                       col_cr='합계(억원)', 
                       col_DP='DP(정제)', 
                       col_PY='PY', 
                       PY=(2018,2022),
                       col_kw='KW1(정제)',
                       cr_n=0):
    """사업명 사전 만드는 함수
    Args:
        df (_type_): _description_
        col_target (str, optional): _description_. Defaults to '사업명(정제)'.
        col_cr (str, optional): _description_. Defaults to '합계(억원)'.
        col_PY (str, optional): _description_. Defaults to 'PY'.
        PY (tuple, optional): _description_. Defaults to (2018,2022).

    Returns:
        _type_: _description_
    2022.12.17(SAT)
    """
    # 조건 :  년도
    df = df[(df[col_PY] >= PY[0]) & (df[col_PY] <= PY[1])].copy() 
    df_temp = make_table_no1(df=df, columns1=[col_DP, col_target], col_kw=col_kw)
    df_temp = df_temp.reset_index() 

    df_temp2 = df_temp[df_temp[col_cr]> cr_n].copy() 
    df_temp2['name']=df_temp2[col_DP] + df_temp2[col_target]
    
    dic_p1 = defaultdict(dict)
    for index, each in df_temp2.iterrows():
        k = each['name']
        for year in range(PY[0],PY[1]+1):
            if year in df_temp2.columns:
                dic_p1[k].update({year:each[year]})
        if col_cr :
            dic_p1[k].update({col_cr:each[col_cr]})
    print(f"* 사업개수(len(dic_p1)) = {len(dic_p1)}")
    return dic_p1

def make_reference_ratio(df=None, df_stage1=None, col_PY='PY', 
                         PY=(2018,2022), 
                         cr_percent=80, cr_fund=100):
    """두 개 데이터 입력받아서, 레프런스 생성한 다음 비교하여 사업별 비율값을 리턴함 
    df : 
    df_stage1 : 
    cr_percent : 전체 사업비총액 대비 최소 몇% 이상 차지하여야 하는가 
    cr_fund : 최소 투자금액
    
    2022.12.17(토)
    """
    print(f">> make_reference_ratio(df, df_stage1, col_PY={col_PY}, PY={PY})")
    
    dic_p1 = make_reference_dic(df, col_PY=col_PY, PY=PY)
    df_r1 = DataFrame(dic_p1).T 
    
    dic_p2 = make_reference_dic(df_stage1, col_PY=col_PY, PY=PY)
    df_r2 = DataFrame(dic_p2).T 
    
    # 머지 
    df_r3 = pd.merge(df_r1, df_r2, how='inner', left_index=True, right_index=True, suffixes=('_total','_sub'))
    df_r3['ratio'] = 100 * df_r3['합계(억원)_sub']/df_r3['합계(억원)_total']
    print(f"* (result 1) merging df et df_stage1 cum how='inner' : total = {df_r3.shape} ")
    
    #cr_percent = 80
    df_r4 = df_r3[ (df_r3['ratio']>= cr_percent) | (df_r3['합계(억원)_sub'] >= cr_fund)].copy()

    name_list_pr1 = df_r4.index.tolist()
    df_r5 = df_r4.reset_index() 
    print(f"* (result 2) compare result : compare = {df_r5.shape} cum cr_percent={cr_percent}%, cr_fund={cr_fund}")
    print(f"* (result 3) name_list program : name_list_pr1 = {len(name_list_pr1)}, {name_list_pr1}")

    return {'total':df_r3,'compare':df_r5, 'name_list':name_list_pr1}
    

#-----------------------------------
# Function Utility for NTIS 
#-----------------------------------
def info(df, col, prefix=''):
    n_total = len(df.index)
    n1 =  df[col].isnull().sum()
    n2 = (~df[col].isnull()).sum()
    r1 = 100 * n1/n_total
    print(f"*{prefix} {col}: Null {r1:.1f} %  ( {n1:,}/{n_total:,} )") 
    
def info2(df1=None, col1='과제고유번호', prefix1='a1', df2=None, col2='과제고유번호', prefix2='a2'):
    info(df1, col1, prefix1)
    info(df2, col2, prefix2)  
    
def compare_value_counts(df1=None, col1='', tag1='a1', df2=None, col2='', tag2='a2'):
    s1 = df1[col1].value_counts() 
    s2 = df2[col2].value_counts() 
    df2 = DataFrame({tag1:s1, tag2:s2})
    df3 = df2.fillna(0)
    df3['total'] = df3[tag1] + df3[tag2]
    df4 = df3.sort_values(by=['total'], ascending=[False])
    return df4

def find_field(df1:DataFrame, df2:DataFrame, field='TI', stopword=r'(코드|가중치)', tag1='a1',tag2='a2')-> dict():
  
        
    '''
    Examples:
        result = bnf.find_field(df1=df_a1, df2=df_a2, field=field, 
                    stopword='코드|가중치|과학기술표준분류코2-중')
        
    '''
    print(f"<<field = {field}>>")
    tag = [tag1, tag2]
    result = {}
    
    for i, df_each in enumerate([df1,df2]):
        print(f'{tag[i]}\n-----')
        temp=[]
        for each in df_each.columns:
            #if field in each:
            if re.search(field, each):
                #print(each)
                if re.search(stopword, each):
                    pass
                else:
                    print(each)
                    temp.append(each)
        result[tag[i]]=temp

        print('='*30)
    return result


def remove_line(x, sep='\n') -> str :
    """ 텍스트 안의 이중 라인을 제거함
    이중 라인 제거한 다음 sep으로 다시 결함하여 리턴함

             
    """
    if x is None:
        return None
    if isinstance(x, float):
        return x
    x2 = x.split('\n')
    x3 = [each for each in x2 if each.strip() != '']
    
    # 엑셀에서 보는 데 불편하여 탭키로 변경(2022.12.14)
    x4 = f'{sep}'.join(x3)

    return x4



def printa(txt, symbol='-', n=50, show=True, kind=2):
    if show : 
        match kind:
            case 1|'one':
                print(symbol*n)
                print(txt) 
            case 2|'two':
                print(symbol*n)
                print(txt) 
                print(symbol*n)

# ----------------------------
#  키워드 개수에 기초하여 중분류 설정하는 함수들 
# ----------------------------



def classify_group(x, ptn_double_list) -> dict:
    """여러개 패턴리스트를 입력 받아서, 그룹1,그룹2 .. 로 태깅을 매김
    패턴에 해당하는 것이  없으면, None 리턴함    
    
    

    2022.12.19(MON)    
    * 내부 호출 함수 count_word_exist() 
    """
    #print(">>> classify_group( )")
    if x is None:
        return None 
    if isinstance(x, float):
        return None
    
    result = {}
    for i, ptnlist in enumerate(ptn_double_list) :
        aa = count_word_exist(x, ptnlist)
        if aa is None:
            pass
        elif isinstance(aa, float):
            print(f"aa = {aa}")
            return None
        else:
            result[f'G{i+1}'] = aa
    if result:
        return result
    else:
        return None 
    

def classify_no(x:dict, kind='single' ) -> str:
    """
    사전 정보를 입력받아서, 사전 크기가 1이며 키 값을 리턴하고, 1보다 크면, 최고값을 지닌 키 중에서 랜덤하게 클래스 번호를 리턴해 줌 
    
    내부호출함수 : find_max_key() 
    """
    if x is None:
        return None
    else:
        if len(x) == 1:
            return ';'.join(list(x.keys()))
        else:
            # 키워드 개수 많은 group 지정해주기,  
            dic = {}
            for k, v in x.items():
                dic[k] = len(v.split(';'))
            return find_max_key(dic, kind=kind) 

        
        
def count_word_exist_from_list(txtlist:list, ptnlist:list, sep=';') -> str:
    """
    입력되는 값이 리스트일 경우 대비함

    """
    x1 = [each.strip() for each in txtlist if each is not None]
    x2 = ' '.join(x1)
    aa = count_word_exist(x=x2, ptnlist=ptnlist)
    return aa

    
    
def count_word_exist(x:str, ptnlist:list, sep=';') -> str:
    """wordlist에 들어 있는 단어들이 and 조건으로 존재하는 건수 구하는 함수
    x : str
    ptnlist : list 
    returns
    -------
    만약 입력한 패턴이 존재하면, 해당하는 패턴을 구분자(디폴트 세미콜론)으로 연결하여 문자열로 만들어 리턴해 줌
    example
    -------
    aa = bnf.count_word_exist('네트워크 기술보안 개발에 대한 연구', 
                              ['기술 보안','네트워크 보안','차세대 보안', '네트워크 보안 기술','개발 연구 보안2'],
                           )
    print(aa)
    기술 보안;네트워크 보안;네트워크 보안 기술
    """
    if x is None:
        return None 
    if isinstance(x, float):
        print(f"{x}")
        return None
    txt = x 
    count_list = []
    for each in ptnlist:
        # 하나의 패턴이 두 개의 단어로 구성되어 있으면, 이들 각각을 구분하여 카운팅을 함
        onoff = False
        count_sub_list =[]
        # 한 개 패턴에 두 개 이상 단어가 존재하면, 각 단어가 몇 번 등장하는 지 카운팅 함
        ptnlist_sub = each.split()
        n_before = len(ptnlist_sub) # 패턴으로 사용할 단어 개수 계산 
        for word in ptnlist_sub:
            temp = re.findall(word, txt, re.IGNORECASE)
            n1 = len(temp)
            if n1>0:
                tag = True 
                count_sub_list.append(word)
            else:
                tag = False
        if len(count_sub_list) == n_before: 
            #print(each)
            count_list.append(each)
    if len(count_list)>0:
        return f'{sep}'.join(count_list)
    else:
        return None 
    
def find_max_key(dic1, kind='single') -> str:
    """사전을 받아서 밸류가 가장 큰 키를 리턴하기.
    만약 밸류가 같은것이 여러 개 있으면,
    kind == single 이면, 랜덤하게 선택해서 하나만 리턴해 주기
    kind == multi  이면, 전체 키 리턴함 
    
    example
    -------
    find_max_key({'G1': 1, 'G2': 2} )
    find_max_key({'G1': 1, 'G2': 1} )
    
    2022.12.19(MON)
    """
    max_v = max(dic1.values())
    max_name = []
    for k, v in dic1.items():
        if v == max_v :
            max_name.append(k)
    if len(max_name)>1:
        
        if kind == 'single':
            # random 하게 선정하기 
            return random.choice(max_name) 
        else:
            return ';'.join(max_name) 
    else:
        return max_name[0]


def get_n_ranked_org(df, col_name, col_value, n=3, org='기관'):
    """
    2023.2.6(M) 상위 n개 를 추출하여 기관명, count, pct 를 알려주는 함수
    col_value : 값을 비교하는 대상이 되는 컬럼명

    """
    df['pct_'] = 100 * df[col_value] / df[col_value].sum()
    df['rank'] = df[col_value].rank(axis=0, ascending=False, method='min')
    df2 = df[df['rank'] <= n]
    txt_list = []
    n_size = df.shape[0]
    if n < n_size:
        n1 = n
    else:
        n1 = n_size

    org1 = trans_word(org, 1)  # org + '는'  # 단어는, 기관은
    org2 = trans_word(org, 2)  # org + '가'  # 단어가, 기관이

    txt = f"전체 {n_size}개 {org} 중에서 상위 {n1}위 이내에 속하는 {org1} 다음과 같이,"
    txt_list.append(txt)
    for index, each in df2.iterrows():
        txt = f"'{each[col_name]}'({each['pct_']:.1f}%, {each[col_value]:,}개)"
        txt_list.append(txt)
        # print(txt)
    txt_list.append('순이다.')

    pct2 = df2['pct_'].sum()
    txt_info = trans_info(pct2)

    txt_list.append(f'이들 {n1}개 {org2} 차지하는 비중은 {txt_info} 전체의 {pct2:.1f}% 이다.')


    txt_c = compare_wordlist(df[col_name].tolist())

    txt_list.append(f'전체 {org} 중에서 정제를 고려해야 할 후보 {org}의 수는 {len(txt_c)} 개 이다. \n 아래는 정제 후보 목록 이다.')
    txt_c2 = '\n - '.join(txt_c)
    txt_list.append(txt_c2)

    #print(' '.join(txt_list))
    #return df2[[col_name, col_value, 'rank']]
    txt_result = ' '.join(txt_list)
    #print(txt_result)
    #return df2[[col_name, col_value, 'rank']]
    return txt_result



# get_n_ranked_org(df=df, col_name='name', col_value='count', n=3, org='단어')
# get_n_ranked_org(df=df, col_name='name', col_value='count', n=3, org='기관')


def trans_word(word, no=1):
    """
    2023.2.6(Mon)
    """
    match no:
        case 1:  # 단어는, 기관은
            if word[-1] == '어':
                return word + '는'
            elif word[-1] == '관':
                return word + '은'
        case 2:  # 단어가, 기관이
            if word[-1] == '어':
                return word + '가'
            elif word[-1] == '관':
                return word + '이'
    return word


def trans_info(value):
    """
    2023.2.6(Mon)"""
    if value > 80:
        txt_super = '거의 대부분을 차지하고 있는 데,'
    elif value > 50:
        txt_super = '과반이상을 차지하는'
    elif value > 30:
        txt_super = '1/3을 상회하는'
    elif value > 10:
        txt_super = ''
    elif value > 5:
        txt_super = ''
    else:
        txt_super = '매우 적은'

    return txt_super


def compare_wordlist(wordlist):
    """
    입력한 단어를 비교하여 부분 집합이 존재하는 지 확인함
    2023.2.6(Mon)
    """
    temp = []

    type1 = '부분집합'
    type2 = '공백' # 공백을 제거하면 같은 기관명
    type3 = '대학교'

    wordlist_a = [each.replace(' ', '') for each in wordlist]
    wordlist1 = wordlist
    for i, w1 in enumerate(wordlist1):
        for j, w2 in enumerate(wordlist1):
            if i > j:
                if (w1 in w2) and (w1 != w2):
                    #temp.append(f"'{w1}', '{w2}' : 비슷한 단어(부분집합관계)")
                    temp.append(f"'{w1}', '{w2}' : {type1}")
                elif w1.replace(' ', '') in w2.replace(' ', ''):
                    #temp.append(f"'{w1}', '{w2}' : 비슷한 단어(공백 제거시 부분집합관계)")
                    temp.append(f"'{w1}', '{w2}' : {type2}")
                else:
                    if '대학교' in w1:
                        w1a = w1.replace('대학교', '대학')
                    else:
                        w1a = w1
                    if '대학교' in w2:
                        w2a = w2.replace('대학교', '대학')
                    else:
                        w2a = w2
                    if w1a in w2a:
                        #temp.append(f"'{w1}', '{w2}' : 비슷한 단어('대학교' 명칭을 '대학'으로 변경시 부분집합관계)")
                        temp.append(f"'{w1}', '{w2}' : {type3}")
    # print(', '.join(temp))
    return temp


# 패턴 제거 할 경우 테이블
def replace_pattern1(pattern1, repl, string1):
    """
    re.sub()가 None 입력하면 에러 발생하기에 None 일 경우 None을 리턴시켜 줄 수 있게 만듦
    """
    if string1 is None:
        return None
    if pattern1 is None:
        return string1
    if isinstance(string1, str) == True:
        return re.sub(pattern1, repl, string1)
    else:
        return string1


def make_table_ptn(df=None,
                           column_index_list=['사업_부처명', '사업명', '내역사업명'],
                           column_index1='사업_부처명',
                           column_index2='사업명',
                           column_search='사업명',
                           column_FUND='정부연구비(원)',
                           column_PY='제출년도',
                           ptn_list=[('','')],
                           show=False,

                           filename=None,
                           dic_column_width={'사업_부처명':20, '사업명':50, '사업수(개)':10,
                                             '부처기준사업수(개)':10, '비교':10,'과제수(개)':10,
                                             '패턴':25,
                                             'memo':30},
                           col_name_float=None,
                           only_summary=True,
                           precision=2
                           ) -> DataFrame() :
    #print(">>>")
    """
    패턴결과표 작성 함수
    :param df:
    :param column_index2:
    :param column_index1:
    :param column_FUND:
    :param column_PY:
    :param ptn_list:  tuple list  (ex) (ptn1, memo1), (ptn2, memo2), ....
    :param show:
    :param col1:
    :param col2:
    :param col3:
    :param col4:
    :param col5:
    :param col6:
    :param filename:
    :param dic_column_width:
    :param only_summary:
    :return:
    2023.6.1(Wed)
    """
    dic_total = dict()

    if df is None:
        return None
    else:
        # 원 데이터 보존하기 위해서,
        df = df.copy()

    column_index2 = column_index_list[1]
    column_index_list2 = column_index_list + [column_PY]

    col0 = 'ptn_no'
    col1 = f'{column_index2}(개)'
    col2 = f'부처기준{column_index2}(개)'
    col3 = '비교'
    col4 = '과제수(개)'
    col5 = '연구비(조원)'
    col6 = 'memo'
    col7 = f'패턴제거시{column_index2}(개)'
    col8 = f'{column_index2}감소수(개)'
    col9 = f'{column_index2}감소비율(%)'

    if filename is not None:
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')

    # 전체 사업 개수
    no_program_total = len(df[column_search].value_counts().index)

    for no_i, each_ptn in enumerate(ptn_list, start=1):
        ptn=each_ptn[0]
        memo=each_ptn[1]
        df1 = df[df[column_search].str.contains(ptn, na=False, case=False)].copy()
        if len(df1.index)>0:
            fund1 = df1[column_FUND].sum() / 1e12
            s1 = df1[column_index2].value_counts()
            # 사업수
            no_program1 = len(s1.index)
            # 과제수
            no_project = s1.sum()
            # 부처기준사업수

            df_dp = df1.groupby(column_index_list2)[column_FUND].agg('sum').unstack(fill_value=0)/100000000  # 억원
            no_program2 = len(df_dp.index)
        else:
            df_dp = DataFrame([])
            no_program1 = 0
            no_program2 = 0
            no_project = 0

        if len(df_dp.index) == 0 :
            print(f"no data : {ptn}, {memo}")
            # 데이터가 없다면 중단하기
            dic1 = {ptn: {col0: no_i,
                          col1: 0,
                          col2: 0,
                          col3: 0,
                          col4: 0,
                          col5: 0,
                          col6: memo,
                          col7: 0,
                          col8: 0,
                          col9: 0 }}
            dic_total.update(dic1)
            # continue
        else:
            if (filename is not None) and (only_summary == False):
                writer = be.make_excel_from_writer_bar(writer=writer,
                                                       df=df_dp,
                                                       sheet_name=f"ptn_{no_i}",
                                                       kind='sheet',
                                                       dic_column_width=dic_column_width,
                                                       col_name_float=col_name_float)

            if show:
                if ptn == '':
                    print(f"*** ptn = '{ptn}' (패턴 없는 경우임)")
                #print(f"* 사업수(개) {c1:,}, 부처기준사업수(개) {c2:,}, 과제수(개) {c3:,}, 연구비(조원) {fund1:.1f}")




            new_column = column_search +'1'
            #df[new_column] = df[column_search].map(lambda x : re.sub(ptn, '', x))
            df[new_column] = df[column_search].map(lambda x: replace_pattern1(ptn, '', x))
            s2 = df[new_column].value_counts()
            no_program3 = len(s2.index)
            dic1 = {ptn: {col0: no_i,
                          col1 : no_program1,
                          col2 : no_program2,
                          col3 : no_program2 - no_program1,
                          col4 : no_project,
                          col5 : fund1,
                          col6 : memo,
                          col7 : no_program3,
                          col8 : no_program_total - no_program3,
                          col9 : 100 * (no_program_total - no_program3)/no_program_total
                          }}
        dic_total.update(dic1)

    df_a = DataFrame(dic_total).T
    df_b = df_a.reset_index()
    df_c = df_b.rename(columns={'index': '패턴'})
    df_c[col1] = df_c[col1].astype(int)
    df_c[col2] = df_c[col2].astype(int)
    df_c[col3] = df_c[col3].astype(int)
    df_c[col4] = df_c[col4].astype(int)

    df_d = df_c[[col0,'패턴', col6, col1, col2, col3, col4, col5, col7, col8, col9]]
    df_d = df_d.sort_values(by=[col0], ascending=[True])
    df_d.index = pd.RangeIndex(start=1, stop=len(df_d.index) + 1)

    if (filename is not None) :
        #df_c2 = df_c.style.format(precision=precision, thousands=',')
        writer = be.make_excel_from_writer_bar(writer=writer,
                                               df=df_d,
                                               sheet_name=f"Summary",
                                               kind='sheet',
                                               dic_column_width=dic_column_width)

        writer.close()

    print(f"shape = {df_d.shape}")
    return df_d


# read file


def read_ntis_files(filenames=None, sheet_name=0) -> DataFrame:
    """
    ntis 조분평 엑셀 파일 이름 리스트를 입력 받아서, dataframe 을 리턴해 줌

    Parameters
    ----------
    * filenames : NTIS 조분평 데이터에서 다운로드 받은 파일 이름 리스트
    * sheet_name : 불러올 쉬트 이름. 디폴트는 0, (ex) '사업과제정보'

    Returns
    -------
    * df

    2023.4.21(Fri)
    2023.4.27(Thr)
    2033.5.31(WED) ntis_find.py 로 옮김

    """
    df_list = []
    for index, each in enumerate(filenames, start=1):
        t1 = time.time()
        #with warnings.catch_warnings(record=True):
        #warnings.simplefilter("always")
        if True:
            xl = pd.ExcelFile(each)
            # sheet names
            s_names = xl.sheet_names
            s_n = len(s_names)

            if s_n >= 1:
                df = pd.read_excel(each, engine="openpyxl", sheet_name=sheet_name)
                #txt=f"No {index} {os.path.basename(each)}(행:{df.shape[0]:,}, 열:{df.shape[1]:,}), 시트 {s_n}개, 시트 이름:{s_names}"
                #print(txt)
                df_list.append(df)
        t2 = time.time() - t1
        print(f"consumed time : {t2: .1f} seconds")

    df = pd.concat(df_list, axis=0)
    return df

if __name__ == '__main__':
	print(">>> BK_NTIS_FIND")
	print(">>> ")