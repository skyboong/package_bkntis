"""
NITS를 분석하는 기본 클래스를 정의한다.
2022.4.13(Wed)
2023.6.19(Mon)

"""
import sys
#sys.path.append("/Users/bk/Dropbox/bkmodule2019/")

import os
import glob
import time

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from matplotlib import cm
import matplotlib.colors as mcolor
from PyPDF2 import PdfMerger 


from . import bk_util as bu
from . import bk_ntis_find as bnf
from . import bk_graph_plotly as bgp

print("Test2 : bk_ntis.py")

col_group = '연구비_등급4'
raw_symbols = SymbolValidator().values

def f_q25(x):
    return x.quantile(q=0.25)

def f_q50(x):
    return x.quantile(q=0.5)

def f_q75(x):
    return x.quantile(q=0.75)

def f_q90(x):
    return x.quantile(q=0.90)

def f_q99(x):
    return x.quantile(q=0.99)

class NTIS():
    """NTIS 과제 분석 클래스
    2022.5.21(금)
    2023.4.27(목)
    2023.5.3(수) 이름변경 NTISproject -> NTIS
    """
    def __init__(self, df=None, filename=None, name_org='해양수산부'):
        """
        데이타프레임으로 입력하거나, 파일로도 입력가능하게 함
        :param df:
        :param filename: 피클파일이어야 함.
        :param name_org:
        """

        t1 = time.time()

        self.df = None

        if df is not None:
            self.df = df.copy()
            print(f"df's shape = {df.shape}")
        elif filename is not None:

            root, ext = os.path.splitext(filename)
            match ext :
                case '.pkl':
                    self.df = pd.read_pickle(filename)
                case '.csv':
                    self.df = pd.read_csv(filename, index_col=0) # 인덱스 읽지 않기
                case _:
                    print('입력 파일이 없군요. 입력 파일 이름을 확인해 주세요')
                    return None
        else:
            pass
        if self.df is not None:
            self.pre_treatment1_fund()
            self.pre_treatment2_name_org(name_org=name_org)
            self.pre_treatment3_region()
            self.pre_treatment4_security()
        t2 = time.time() - t1
        print(f"*** NTIS 객체 생성 : {t2:.1f} 초")

    def __repr__(self):
        class_name = type(self).__name__
        return f"class name : {class_name}"

    def __len__(self):
        if self.df is not None:
            return len(self.df.index)
        else:
            return 0

    def filtering(self, id_list=[], colname='과제고유번호'):
        """필터링 부문"""
        if self.df is not None:
            print("<<필터링 결과>>")
            print(f"* 입력한 과제고유번호 개수는 {len(id_list):,}개이며, 이중에서 유니크한 것은 {len(set(id_list)):,}개 이다.")
            onoff1 = self.df[colname].isin(id_list)
            df1 = self.df[onoff1].copy()
            print(f"* 전체 과제수는 {self.df.shape[0]:,}개이며, 입력한 {colname} 리스트를 적용하여 {df1.shape[0]:,}개로 필터링하였다.")
            set1 = set(id_list) - set(self.df[colname].tolist())
            print(f"* 전체 과제에 포함되지 않은 입력한 과제번호 개수는 {len(set1):,}개 이다.")
            return df1

    def import_file(self, filename=None):
        """파일 불러오는 메소드"""
        if filename is not None:
            root, ext = os.path.splitext(filename)
            match ext :
                case '.pkl':
                    df = pd.read_pickle(filename)
                case '.csv':
                    df = pd.read_csv(filename, index_col=0) # 인덱스 읽지 않기
                case _:
                    df = None
                    print('입력 파일이 없군요. 입력 파일 이름을 확인해 주세요')
            return df

    def import_performance_data(self, filename=None, df=None, kind=1):
        if df is not None:
            self.df_performance = df
        else:
            if kind == 1 :
                df = self.import_file(filename=filename)
                if df is not None:
                    self.df_performance = df
                    print(f"* imported {self.df_performance.shape[0]:,}개 임포트 완료하였습니다.")
                    df_raw1 = df
                    # 보안과제 정보 알려주기
                    ## 체크 : 보안과제 제외, 전체 논문수 대비 보안과제 논문의 비율 체크

                    a1 = len(df_raw1.index)

                    onoff1 = df_raw1['성과발생년도'] != '보안과제의 성과정보'
                    a2 = onoff1.sum()

                    onoff_s = df_raw1['성과발생년도'] == '보안과제의 성과정보'
                    a3 = onoff_s.sum()

                    # 3116
                    # 8437

                    ## 체크 : 기여율, 기여율 0 제외

                    onoff2 = df_raw1['기여율(확정)'] == 0
                    a4 = onoff2.sum()

                    onoff3 = df_raw1['기여율(확정)'] > 0
                    a5 = onoff3.sum()

                    # 보안과제가 아니고, 기여율도 0 초과하는 과제수는 ?
                    onoff = onoff1 & onoff3
                    a6 = onoff.sum()

                    # 307513
                    # 730741

                    print(
                        f"* 전체 논문({a1:,}건) 중 보안과제의 논문 {a3:,}건({100 * a3 / a1:.1f}%)과 "
                        f"기여율 0% 논문 {a4:,}건({100 * a4 / a1:.1f}%)을 "
                        f"제외한 논문 {a5:,}건({100 * a5 / a1:.1f}%)을 분석 대상으로 하였다.")

                    self.df_performance2 = df_raw1[onoff].copy()

                else:
                    self.df_performance = None
                    print("* imported None")
            else:
                self.df_performance = None




    def show_info(self):
        print("shape = ", self.df.shape)
        print(f"memory usage =  {self.df.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB")

    def pre_treatment1_fund(self):
        """작업 편의를 위해서 전처리함"""
        txt = ">> pre_treatment1_fund() : (컬럼생성) 정부연구비_억, 정부연구비_조, 연구비_등급 1/2/3/4, 보안과제"
        print(txt)

        df_raw = self.df
        df_raw['정부연구비_억'] = df_raw['정부연구비(원)'] / 1e8  # 억원
        df_raw['정부연구비_조'] = df_raw['정부연구비(원)'] / 1e12  # 조원

        def set_class(x):
            if x >= 5 * 1e8:
                name = 'A(5억원이상)'
            elif x >= 2 * 1e8:
                name = 'B(2~5억원미만)'
            elif x >= 1e8:
                name = 'C(1~2억원미만)'
            elif x >= 5e7:
                name = 'D(5천만~1억원미만)'
            elif x >= 3e7:
                name = 'E(3천만~5천만미만)'
            else:
                name = 'F(3천만미만)'
            return name

        def set_class8(x):
            """구간"""
            if x >= 10 * 1e9:
                name = 'A(100억원이상)'
            elif x >= 5 * 1e8:
                name = 'B(5~100억원미만)'
            elif x >= 2 * 1e8:
                name = 'C(2~5억원미만)'
            elif x >= 1e8:
                name = 'D(1~2억원미만)'
            elif x >= 5e7:
                name = 'E(5천만~1억원미만)'
            elif x >= 3e7:
                name = 'F(3천만~5천만원미만)'
            elif x >= 1e7:
                name = 'G(1천만~3천만원미만)'
            elif x >= 1e6:
                name = 'H(100만~1천만원미만)'
            else:
                name = 'I(100만원미만)'
            return name
        df_raw['연구비_등급1'] = df_raw['정부연구비(원)'].map(lambda x: set_class(x))
        df_raw['연구비_등급2'] = df_raw['정부연구비(원)'].map(lambda x: set_class8(x))

        bins1 = [0, 1e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10, 5e10, 1e11, np.inf]
        labels1 = ['100만원 미만',
                   '100<= <10000만원',
                   '1000<= <5000만원',
                   '5000만원<=  <1억원',
                   '1억원<= <5억원',
                   '5억<= <10억원',
                   '10억<= <50억원',
                   '50억<= <100억원',
                   '100억원<= <500억원',
                   '500억원<= <1000억원',
                   '1000억원 이상']

        bins2 = [0, 1e8, 1e9, 1e10, 1e11, np.inf]
        labels2 = ['1억원 미만',
                   '10억원 미만',
                   '100억원 미만',
                   '1000억원 미만',
                   '1000억원 이상']

        df_raw['연구비_등급3'] = pd.cut(df_raw['정부연구비(원)'],
                                   bins=bins1,
                                   labels=labels1,
                                   right=False
                                   )
        df_raw['연구비_등급4'] = pd.cut(df_raw['정부연구비(원)'],
                                   bins=bins2,
                                   labels=labels2,
                                   right=False
                                   )

        bins3 = [0, 1e8, 5*1e9, np.inf]
        labels3 = ['~1억원 미만',
                   '1억원 이상 ~ 50억원 미만',
                   '50억원 이상',
                   ]
        df_raw['연구비_등급5'] = pd.cut(df_raw['정부연구비(원)'],
                                   bins=bins3,
                                   labels=labels3,
                                   right=False
                                   )



        self.df = df_raw

    def pre_treatment2_name_org(self, name_org='해양수산부'):
        """
        사업 부처명 정리
        :return:
        """
        txt = ">> pre_treatment2_name_org() : 사업 부처명 정리 (컬럼생성)  사업_부처명2/3/4/m"
        print(txt)

        dic1 = {'미래창조과학부': '과학기술정보통신부',
                '중소기업청': '중소벤처기업부',
                '국무총리실': '국무조정실',
                '범부처 사업': '다부처'}

        dic_b = {'미래창조과학부': '과학기술정보통신부',
                 '국무총리실': '국무조정실',
                 '농림수산식품부': '농림축산식품부',
                 '국민안전처': '행정안전부',
                 '범부처 사업': '다부처',
                 '소방방재청': '소방청',
                 '식품의약품안전청': '식품의약품안전처',
                 '중소기업청': '중소벤처기업부',
                 '지식경제부': '산업통상자원부',
                 '외교통상부': '외교부',
                 '행정자치부': '행정안전부',
                 '안전행정부': '행정안전부',
                 }

        dic_m = {'미래창조과학부': '과학기술정보통신부',
                 '범부처 사업': '다부처',
                 '중소기업청': '중소벤처기업부',
                 '지식경제부': '산업통상자원부',
                 '교육과학기술부': '과학기술정보통신부',
                 '국토해양부': '국토교통부',
                 }

        etc_dept = ['국민안전처', '행정안전부', '행정자치부', '고용노동부', '새만금개발청', '인사혁신처',
                    '통일부', '공정거래위원회', '행정중심복합도시건설청', '법무부', '외교부',
                    '여성가족부', '법제처', '기획재정부', '소방청', '경찰청',
                    '해양경찰청', '특허청', '문화재청', '국방부',
                    '식품의약품안전처', '문화체육관광부', '원자력안전위원회']

        etc_dept2 = ['국민안전처', '행정안전부', '행정자치부', '고용노동부', '새만금개발청', '인사혁신처',
                     '통일부', '공정거래위원회', '행정중심복합도시건설청', '법무부', '외교부',
                     '여성가족부', '법제처', '기획재정부', '소방청', '경찰청',
                     '해양경찰청', '특허청', '문화재청', '국방부',
                     '식품의약품안전처', '문화체육관광부', '원자력안전위원회']

        etc_dept_m = ['국가과학기술위원회', '국가청소년위원회',
                      '국민안전처', '안전행정부', '행정안전부', '행정자치부',
                      '국무총리실', '국무조정실',
                      '고용노동부', '새만금개발청', '인사혁신처',
                      '농림축산식품부', '농림수산식품부',
                      '통일부', '공정거래위원회', '행정중심복합도시건설청', '법무부', '외교부', '외교통상부',
                      '여성가족부', '법제처', '기획재정부',
                      '소방방재청', '소방청', '경찰청',
                      '기상청',
                      '산림청',
                      '해양경찰청', '특허청', '문화재청', '국방부', '방송통신위원회',
                      '식품의약품안전처', '식품의약품안전청',
                      '문화체육관광부', '원자력안전위원회']

        def change_name(x, dic2, etc):
            if x in dic2.keys():
                return dic2[x]
            elif x in etc:
                return '기타부청'
            else:
                return x

        def change_name3(x='', name_org='해양수산부'):
            if x in [name_org]:
                return name_org
            else:
                return '나머지 부처'

        dfc = self.df
        dfc['사업_부처명2'] = dfc['사업_부처명'].map(lambda x: change_name(x, dic2=dic1, etc=etc_dept))
        dfc['사업_부처명3'] = dfc['사업_부처명'].map(lambda x: change_name3(x, name_org=name_org))

        #  1999년 부처명 부터 정리함
        #  지시경제부 -> 산업통상자원부
        # dfc['사업_부처명4'] = dfc['사업_부처명'].map(lambda x: change_name(x=x, dic2=dic_b, etc=[]))
        dfc['사업_부처명4'] = dfc['사업_부처명'].map(lambda x: change_name(x=x, dic2=dic_b, etc=etc_dept2))

        # major 부처명 표기
        dfc['사업_부처명m'] = dfc['사업_부처명'].map(lambda x: change_name(x=x, dic2=dic_m, etc=etc_dept_m))
        print(f"-- 사업_부처명m : NTIS 에 입력되어 있는 사업_부처명은 2012년 기준으로 되어 있음. 이것을 2020년 기준으로 변경함{dic_m}")

        self.df = dfc

    def pre_treatment3_region(self):
        """
        지역명 정리
        :return:
        """
        txt = ">> pre_treatment3_region() : 지역명 정리, (컬럼생성) 지역2, 지역3"
        print(txt)
        df_raw = self.df

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

        def change_region(x, dic1):
            if x in dic1.keys():
                return dic1[x]
            else:
                return x

        df_raw['지역2'] = df_raw['지역'].map(lambda x: change_region(x, dic1=dic_region2))

        # 지역3 는 수도권, 겨앙권, 전라권,충청권, 제주, 강원으로 구분
        df_raw['지역3'] = df_raw['지역'].map(lambda x: change_region(x, dic1=dic_region3))

        self.df = df_raw

    def pre_treatment4_security(self):
        """
        보안과제 찾기 : 1) 내역사업명에 '보안과제정보', 2) 과제명에 '보안과제정보'
        :return:
        """
        print(">> pre_treatment4_security() : (컬럼생성) 보안과제2, 보안과제3")

        df_raw = self.df
        # 보안과제
        # 내역사업명에 '보안과제정보' 라는 단어가 들어가 있는 경우 검토
        df_raw['보안과제'] = df_raw['내역사업명'].str.contains('보안과제정보', na=False).copy()
        df_raw['보안과제2'] = df_raw['보안과제'].map(lambda x: '보안' if x else '일반')
        print("(참고 1) 보안과제2 : ", df_raw['보안과제2'].value_counts())
        df_raw['보안과제3'] = df_raw['과제명'].map(lambda x: '보안' if x == '보안과제정보' else '일반')
        print("(참고 2) 보안과제3 : ", df_raw['보안과제3'].value_counts())

        self.df = df_raw

    def plot_paper1(self,
                    figsize=(800,600),
                    PY1='성과발생년도',
                    PY2='과제수행년도',
                    col_1 = '기여율(확정)',
                    range_l=None, range_r=None):

        df_raw = self.df_performance


        #
        df_raw['col_100']=df_raw[col_1]/100.
        df_table1 = df_raw.groupby([PY2])['col_100'].agg(['sum'])

        return df_table1

        '''


        if range_l is None:
            l_min = df_table1['sum'].min()
            l_max = df_table1['sum'].max() * 1.1
            range_l = [l_min, l_max]

        if range_r is None:
            r_min = df_table1['count'].min()
            r_max = df_table1['count'].max() * 1.5
            range_r = [r_min, r_max]

        fig = bgp.make_scatter_and_line3(df=df_table1,
                                      col_left='sum',
                                      col_right='count',
                                      title_yaxes_left='집행액',
                                      title_yaxes_right='과제수',
                                      unit1=' 조원',
                                      unit2=' 개',
                                      precision1=1,
                                      precision2=0,
                                      textfont_size1=8,
                                      textfont_size2=5,
                                      marker_size=10,
                                      marker_width=1,
                                      textposition1='outside',
                                      textposition2='bottom right',
                                      range1=range_l,
                                      range2=range_r,
                                      title=title1,
                                      title_font_size=title_font_size,
                                      cagr=True,
                                      text_cagr_font_size=text_cagr_font_size,
                                      width_string=width_string,
                                      figsize=figsize)
        #fig_list.append(fig)
        return fig
        '''




    def plot_1(self,df=None,
                col_fund1='정부연구비_조',  title1="정부 연구개발 집행액과 과제수",
                col_fund2='정부연구비_억',  title2="정부 연구개발 집행액(평균, 중앙값)" ,
                colname=['mean', 'median'],
                precision=1,
                node_size=3,
                marker_line_width=1,
                marker_color="#ffffff",
                line_width=3,
                output='screen',
                filename='',
                PY='제출년도',
                range_l=None, range_r=None, color_map='Set3', figsize=(800,600),
                title_font_size=20, text_cagr_font_size=15, width_string=200):
        """
        plot_1a()
        """
        if df is None:
            df_raw = self.df
        else:
            df_raw = df.copy()

        # 조원
        df_table1 = df_raw.groupby([PY])[col_fund1].agg(['count', 'sum'])
        if range_l is None:
            l_min = df_table1['sum'].min()
            l_max = df_table1['sum'].max() * 1.1
            range_l = [l_min, l_max]

        if range_r is None:
            r_min = df_table1['count'].min()
            r_max = df_table1['count'].max() * 1.5
            range_r = [r_min, r_max]

        fig_list = []
        fig = bgp.make_scatter_and_line3(df=df_table1,
                                      col_left='sum',
                                      col_right='count',
                                      title_yaxes_left='집행액',
                                      title_yaxes_right='과제수',
                                      unit1=' 조원',
                                      unit2=' 개',
                                      precision1=1,
                                      precision2=0,
                                      textfont_size1=8,
                                      textfont_size2=5,
                                      marker_size=10,
                                      marker_width=1,
                                      textposition1='outside',
                                      textposition2='bottom right',
                                      range1=range_l,
                                      range2=range_r,
                                      title=title1,
                                      title_font_size=title_font_size,
                                      cagr=True,
                                      text_cagr_font_size=text_cagr_font_size,
                                      width_string=width_string,
                                      figsize=figsize)
        fig_list.append(fig)


        df_table = df_raw.groupby([PY])[col_fund2].agg(
            ['count', 'sum', 'mean', 'median', np.min, np.max, f_q90, f_q99])
        fig = bgp.make_graph_line(df=df_table[colname].T,
                               title=title2,
                               title_font_size=title_font_size,
                               unit='',
                               precision=precision,
                               yaxes_title='집행액(억원)',
                               color_map=color_map,
                               cagr=True,
                               text_cagr_font_size=text_cagr_font_size,
                               width_string=width_string,
                               node_size=node_size,
                               marker_line_width=marker_line_width,
                               marker_color=marker_color,
                               line_width=line_width,
                               figsize=figsize)
        fig_list.append(fig)

        if output == 'screen':
            for fig in fig_list:
                fig.show()
        elif output == 'file':
            for i, fig in enumerate(fig_list, start=1):
                fn = filename + f'_{i:02d}.pdf'
                fig.write_image(fn)



    def plot_1b(self, col_fund='정부연구비_억', output='screen', filename='', PY='제출년도', title="년도별 과제당 집행액 추이",
                colname=['mean','median'], precision=1, color_map='tab20c', figsize=(800,600), title_font_size=20,
                text_cagr_font_size=10,
                width_string=200, node_size=8, line_width=2):
        """
        plot_1b()
        """
        df_raw = self.df
        df_table = df_raw.groupby([PY])[col_fund].agg(
            ['count', 'sum', 'mean', 'median', np.min, np.max, f_q90, f_q99])
        fig = bgp.make_graph_line(df=df_table[colname].T,
                               title=title,
                               title_font_size=title_font_size,
                               unit='',
                               precision=precision,
                               yaxes_title='집행액(억원)',
                               color_map=color_map,
                               cagr=True,
                               text_cagr_font_size=text_cagr_font_size,
                               width_string=width_string,
                               node_size=node_size,
                               marker_line_width=1,
                               marker_color="#ffffff",
                               line_width=line_width,
                               figsize=figsize)
        if output == 'screen':
            fig.show()
        elif output == 'file':
            fn =  filename + '.pdf'
            fig.write_image(fn)
            print(f"{fn} file was saved.")
        #return fig


    def plot_2_old(self,
               col_group='연구비_등급1', col_fund='정부연구비_조', col_fund2='정부연구비_억',
               output='screen', filename='',
               PY='제출년도',
               fig_no=1, year1=2018, year2=2021, width=0.2, height=1000,
               figsize=(800,600), plot_bgcolor='rgba(230,236,245)', color_map='Set3',
               title_font_size=20,
               text_cagr_font_size=10,
               width_string=200,
               df_null_count=None):
        """
        연구비 등급1은 NTIS 분석 보고서에 등장하는 기준 준용

        :param col_group: 연구비_등급1/2/3/4, 보안과제, 사업_부처명4
        :param col_fund: 정부연구비_조,
        :return:
        """

        df_raw = self.df
        fig_list = []

        # null data 현황
        if df_null_count is not None:
            df_g = df_null_count.loc[col_group, :]
            fig0 = bgp.make_graph_bar2(df_g,
                                  width=0.5,
                                  title=f'{fig_no}-1 {col_group} 널 데이터 현황(null data) ',
                                  precision=1,
                                  orientation='v',
                                  # height=1800,
                                  tickfont_size=5,
                                  unit=" % "
                                  # yaxes_title=f'널 비율(%) '
                                  )
            #fig.write_image(prefix + f'_null_{col_group1}.pdf')
            fig_list.append(fig0)



        # 과제건수
        df_a1 = df_raw.groupby([col_group, PY])[PY].agg('count').unstack(fill_value=0)
        df_a = df_a1[sorted(df_a1.columns)]  # 컬럼이 정렬되지 않는 문제가 발생함 2022.5.26

        # 연구비
        df_b1 = df_raw.groupby([col_group, PY])[col_fund].agg('sum').unstack(fill_value=0)
        df_b = df_b1[sorted(df_b1.columns)]

        # 평균
        df_c = 10000 * df_b / df_a # 조원에서 억원으로 변경하기
        df_c2 = df_c.fillna(0)



        fig = bgp.make_graph_line(df=df_a,
                                  title=f"{fig_no}-2 {col_group} 과제수 추이",
                                  title_font_size=title_font_size,
                                  xaxes_title='연도',
                                  yaxes_title='과제수',
                                  node_size=8,
                                  mode='lines+markers',
                                  line_width=3,
                                  marker_color="#ffffff",
                                  color_map=color_map, # 'Set3',  # 'tab20c',
                                  plot_bgcolor=plot_bgcolor,
                                  cagr=True,
                                  text_cagr_font_size=text_cagr_font_size,
                                  width_string=width_string,
                                  figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_line(df=df_b,
                                  title=f"{fig_no}-3 {col_group} 연구비 추이",
                                  title_font_size=title_font_size,
                                  xaxes_title='연도',
                                  yaxes_title='연구비(조원)',
                                  node_size=8,
                                  mode='lines+markers',
                                  line_width=3,
                                  marker_color="#ffffff",
                                  color_map=color_map,  # 'tab20c',
                                  plot_bgcolor=plot_bgcolor,
                                  cagr=True,
                                  text_cagr_font_size=text_cagr_font_size,
                                  width_string=width_string,
                                  figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_line(df=df_c2,
                                   title=f"{fig_no}-4 {col_group} 추이 (평균)",
                                   title_font_size=title_font_size,
                                   xaxes_title='연도',
                                   yaxes_title='연구비 평균(억원)',
                                   node_size=8,
                                   mode='lines+markers',
                                   line_width=3,
                                   marker_color="#ffffff",
                                   color_map=color_map,  # 'tab20c',
                                   plot_bgcolor=plot_bgcolor,
                                   cagr=True,
                                   text_cagr_font_size=text_cagr_font_size,
                                   width_string=150,
                                   figsize=figsize)
        fig_list.append(fig)






        fig = bgp.make_graph_bar2(df=df_a.T,
                               title=f"{fig_no}-5 {col_group} 과제수 추이(Bar) ",
                               title_font_size=title_font_size,
                               unit=' ',
                               precision=0,
                               yaxes_title='과제수',
                               xaxes_title='연도',
                               color_map=color_map, # 'gnuplot',
                               textfont_size=8,
                               to_percent=False,
                               figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_b.T,
                               title=f"{fig_no}-6 {col_group} 연구비 추이(Bar)",
                               title_font_size=title_font_size,
                               unit=' ',
                               precision=1,
                               yaxes_title='정부연구비',
                               xaxes_title='연도',
                               color_map=color_map, # 'gnuplot',
                               textfont_size=8,
                               to_percent=False,
                               figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_a.T,
                               title=f"{fig_no}-7 {col_group} 과제수 비율 추이(Bar)",
                               title_font_size=title_font_size,
                               unit='%',
                               precision=1,
                               yaxes_title='비율(%)',
                               xaxes_title='연도',
                               color_map=color_map, # 'gnuplot',
                               textfont_size=8,
                               to_percent=True,
                               figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_b.T,
                               title=f"{fig_no}-8 {col_group} 연구비 비율 추이(Bar)",
                               title_font_size=title_font_size,
                               unit=' ',
                               precision=1,
                               yaxes_title='비율(%)',
                               xaxes_title='연도',
                               color_map=color_map, #'gnuplot',
                               textfont_size=8,
                               to_percent=True,
                               figsize=figsize)
        fig_list.append(fig)

        # sum 억원기준, year1, year2 로 제한하여 나타내기
        unit1 = '억원'
        unit2 = '개'
        df_table = df_raw.groupby([col_group, PY])[col_fund2].agg('sum').unstack(fill_value=0)
        s_sum1 = df_table.sum(axis=1)
        df_table_no1 = df_table.loc[:, year1:year2]
        df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]

        fig = bgp.make_graph_bar2(df=df_table_no2,
                                   orientation='h',
                                   barmode='group',
                                   width=width,
                                   height=height,
                                   opacity=0.8,
                                   reversed=True,
                                   unit=unit1,
                                   title=f"{fig_no}_9 {col_group} 집행액",
                                   title_font_size=title_font_size,
                                   yaxes_title='집행액(억원)',
                                   color_map=color_map,
                                   figsize=figsize)
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_table_no2,
                                  orientation='h',
                                  barmode='stack',
                                  width=width,
                                  height=height,
                                  opacity=0.8,
                                  reversed=True,
                                  unit=unit1,
                                  title=f"{fig_no}_10 {col_group} 집행액",
                                  title_font_size=title_font_size,
                                  yaxes_title='집행액(억원)',
                                  color_map=color_map,
                                  figsize=figsize)
        fig_list.append(fig)

        # ------------------
        # Total :
        # ------------------

        fig = bgp.make_pie(values=s_sum1,
                           labels=s_sum1.index,
                           text1=f"Total(집행액)",
                           unit=unit1,
                           # pull_index=[6],
                           # pull_index_ratio=0.1,
                           # pull_index_ratio_default=0.01,
                           title=f"{fig_no}-11 {col_group} 집행액)",
                           title_font_size=title_font_size,
                           colormap_name=color_map,  # "tab20c",
                           figsize=figsize)
        fig_list.append(fig)



        if year1 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year1],
                            labels=df_table_no2.index,
                            text1=f"{year1}",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-12 {col_group} 집행액({year1})",
                            title_font_size=title_font_size,
                            colormap_name=color_map,  # "tab20c",
                            figsize=figsize)
            fig_list.append(fig)

        if year2 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year2],
                            labels=df_table_no2.index,
                            text1=f"{year2}",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-13 {col_group} 집행액({year2})",
                            title_font_size=title_font_size,
                            colormap_name=color_map,  # "tab20c",
                            figsize=figsize)
            fig_list.append(fig)

        # count
        df_table = df_raw.groupby([col_group, PY])[col_fund].agg('count').unstack(fill_value=0)
        s_sum1 = df_table.sum(axis=1)
        df_table_no1 = df_table.loc[:, year1:year2]
        df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]


        fig = bgp.make_graph_bar2(df=df_table_no2,
                                   orientation='h',
                                   barmode='group',
                                   width=width,
                                   height=height,
                                   opacity=0.8,
                                   reversed=True,
                                   unit=unit2,
                                   title=f"{fig_no}-14 {col_group} 과제수",
                                   title_font_size=title_font_size,
                                   yaxes_title='과제수',
                                   color_map=color_map,
                                   figsize=figsize)
        fig_list.append(fig)

        # ------------------
        # Total :
        # ------------------
        fig = bgp.make_pie(values=s_sum1,
                           labels=s_sum1.index,
                           text1=f"Total(과제수)",
                           unit=unit2,
                           # pull_index=[6],
                           # pull_index_ratio=0.1,
                           # pull_index_ratio_default=0.01,
                           title=f"{fig_no}-15 {col_group} 과제수",
                           title_font_size=title_font_size,
                           colormap_name=color_map, # "tab20c",
                           figsize=figsize)
        fig_list.append(fig)

        if year1 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year1],
                            labels=df_table_no2.index,
                            text1=f"{year1}",
                            unit=unit2,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-16 {col_group} 과제수({year1})",
                            title_font_size=title_font_size,
                            colormap_name=color_map, # "tab20c",
                            figsize=figsize)
            fig_list.append(fig)

        if year2 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year2],
                            labels=df_table_no2.index,
                            text1=f"{year2}",
                            unit=unit2,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-17 {col_group} 과제수({year2})",
                            title_font_size=title_font_size,
                            colormap_name=color_map,  # "tab20c",
                            figsize=figsize)
            fig_list.append(fig)


        #fig_list = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13]

        if output == 'screen':
            for fig in fig_list:
                fig.show()
        elif output == 'file':
            for i, fig in enumerate(fig_list, start=1):
                fn = filename + f'_{i:02d}.pdf'
                fig.write_image(fn)
                #print(f"{fn} was saved")

    def plot_2(self,
               col_group='연구비_등급1', col_fund='정부연구비_조', col_fund2='정부연구비_억',
               output='screen', filename='',
               PY='제출년도',
               fig_no=1, year1=2018, year2=2021, width=0.2, height=1000,
               figsize=(800,600), plot_bgcolor='rgba(230,236,245)', color_map='Set3',
               title_font_size=20,
               text_cagr_font_size=10,
               width_string=200,
               df_null_count=None,
               include_count_sum='both'):
        """
        연구비 등급1은 NTIS 분석 보고서에 등장하는 기준 준용

        :param col_group: 연구비_등급1/2/3/4, 보안과제, 사업_부처명4
        :param col_fund: 정부연구비_조,
        :return:

        2023.6.13 수정
        """

        df_raw = self.df
        fig_list = []

        # null data 현황
        if df_null_count is not None:
            df_g = df_null_count.loc[col_group, :]
            fig0 = bgp.make_graph_bar2(df_g,
                                  width=0.5,
                                  title=f'{fig_no}-0 {col_group} 널 데이터 현황(null data) ',
                                  precision=1,
                                  orientation='v',
                                  # height=1800,
                                  tickfont_size=5,
                                  unit=" % "
                                  # yaxes_title=f'널 비율(%) '
                                  )
            #fig.write_image(prefix + f'_null_{col_group1}.pdf')
            fig_list.append(fig0)

        if include_count_sum in ['money', 'both'] :
            fig_list1 = self.plot_2a(col_group=col_group,
                    col_fund=col_fund,
                    col_fund2=col_fund2,
                    output=output,
                    filename_prefix=filename,
                    PY=PY,
                    fig_no=f"{fig_no}m", year1=year1, year2=year2, width=width, height=height,
                    figsize=figsize, plot_bgcolor=plot_bgcolor, color_map=color_map,
                    title_font_size=title_font_size,
                    text_cagr_font_size=text_cagr_font_size,
                    width_string=width_string,
                    agg_function='sum',  fig_start_no=1)
            fig_list.extend(fig_list1)

        if include_count_sum in ['count', 'both']:
            no_fig_list = len(fig_list)
            fig_list2 = self.plot_2a(col_group=col_group,
                                 col_fund=col_fund,
                                 col_fund2=col_fund2,
                                 output=output,
                                 filename_prefix=filename,
                                 PY=PY,
                                 fig_no=f"{fig_no}c", year1=year1, year2=year2, width=width, height=height,
                                 figsize=figsize, plot_bgcolor=plot_bgcolor, color_map=color_map,
                                 title_font_size=title_font_size,
                                 text_cagr_font_size=text_cagr_font_size,
                                 width_string=width_string,
                                 agg_function='count', fig_start_no=1) #no_fig_list+1)
            fig_list.extend(fig_list2)

        # file name 이 문자열로 입력되어 있다면 파일로 저장해 주기
        if (output == 'file') or (filename != ''):
            for i, fig in enumerate(fig_list, start=1):
                fn = filename + f'_{i:02d}.pdf'
                fig.write_image(fn)
                #print(f"{fn} was saved")
        elif output == 'screen':
            for fig in fig_list:
                fig.show()
        return fig_list


    def plot_2a(self,
                col_group='연구비_등급1',
                col_fund='정부연구비_조',
                col_fund2='정부연구비_억',
                output='screen',
                filename_prefix='',
                PY='제출년도',
                fig_no=1, year1=2018, year2=2021, width=0.2, height=1000,
                figsize=(800,600), plot_bgcolor='rgba(230,236,245)', color_map='Set3',
                title_font_size=20,
                text_cagr_font_size=10,
                width_string=200,
                df_null_count=None,
                agg_function ='sum',
                fig_start_no = 1):
        """
        2023.6.12(Monday)
        """

        df_raw = self.df
        fig_list = []


        # 연구비 기준
        df_money1 = df_raw.groupby([col_group, PY])[col_fund].agg('sum').unstack(fill_value=0)
        df_money = df_money1[sorted(df_money1.columns)]

        # 과제건수 또는 연구비 기준
        df_count1 = df_raw.groupby([col_group, PY])[PY].agg('count').unstack(fill_value=0)
        df_count = df_count1[sorted(df_count1.columns)]  # 컬럼이 정렬되지 않는 문제가 발생함 2022.5.26

        # 평균
        if agg_function == 'sum':
            name1 = '연구비'
            df_c = 10000 * df_money / df_count  # 과제당 투자액(조원에서 억원으로 변경하기)
            df_c2 = df_c.fillna(0)
            df_graph = df_money

            # sum 억원기준, year1, year2 로 제한하여 나타내기
            unit1 = '억원'
            unit2 = '조원'
            memo1 = '과제당 연구비(억원)'

            df_table = df_raw.groupby([col_group, PY])[col_fund2].agg(agg_function).unstack(fill_value=0)

            df_table_no1 = df_table.loc[:, year1:year2]
            df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]
            s_sum1 = df_table.sum(axis=1)
            s_sum2 = df_table_no2.sum(axis=1)
        else:
            name1 = '과제수'

            df_c = df_count / (10000 * df_money)  # 연구비 당 과제수 (1억 투자하면 몇개 과제 지원하는가 ?)
            df_c2 = df_c.fillna(0)
            df_graph = df_count
            # count
            unit1 = '개'
            unit2 = '개'
            memo1 = '1억당 과제수(개)'
            df_table = df_raw.groupby([col_group, PY])[col_fund].agg(agg_function).unstack(fill_value=0)
            df_table_no1 = df_table.loc[:, year1:year2]
            df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]
            s_sum1 = df_table.sum(axis=1)
            s_sum2 = df_table_no2.sum(axis=1)



        fig = bgp.make_graph_line(df=df_graph,
                                  title=f"{fig_no}-{fig_start_no} {col_group} {name1} 추이",
                                  title_font_size=title_font_size,
                                  xaxes_title='연도',
                                  yaxes_title=f"{name1}({unit2})",
                                  node_size=8,
                                  mode='lines+markers',
                                  line_width=3,
                                  marker_color="#ffffff",
                                  color_map=color_map,  # 'tab20c',
                                  plot_bgcolor=plot_bgcolor,
                                  cagr=True,
                                  text_cagr_font_size=text_cagr_font_size,
                                  width_string=width_string,
                                  figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        fig = bgp.make_graph_line(df=df_c2,
                                   title=f"{fig_no}-{fig_start_no} {col_group} {memo1}",
                                   title_font_size=title_font_size,
                                   xaxes_title='연도',
                                   yaxes_title=f'{name1}({unit1})',
                                   node_size=8,
                                   mode='lines+markers',
                                   line_width=3,
                                   marker_color="#ffffff",
                                   color_map=color_map,  # 'tab20c',
                                   plot_bgcolor=plot_bgcolor,
                                   cagr=True,
                                   text_cagr_font_size=text_cagr_font_size,
                                   width_string=150,
                                   figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_graph.T,
                               title=f"{fig_no}-{fig_start_no} {col_group} {name1} 추이(Bar)",
                               title_font_size=title_font_size,
                               unit=' ',
                               precision=1,
                               yaxes_title=f'{name1}({unit2})',
                               xaxes_title='연도',
                               color_map=color_map, # 'gnuplot',
                               textfont_size=8,
                               to_percent=False,
                               figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_graph.T,
                               title=f"{fig_no}-{fig_start_no} {col_group} {name1} 비율 추이(Bar)",
                               title_font_size=title_font_size,
                               unit=' ',
                               precision=1,
                               yaxes_title='비율(%)',
                               xaxes_title='연도',
                               color_map=color_map, #'gnuplot',
                               textfont_size=8,
                               to_percent=True,
                               figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_table_no2,
                                   orientation='h',
                                   barmode='group',
                                   width=width,
                                   height=height,
                                   opacity=0.8,
                                   reversed=True,
                                   unit=unit1,
                                   title=f"{fig_no}_{fig_start_no} {col_group} {name1} 추이 (group)",
                                   title_font_size=title_font_size,
                                   yaxes_title=f'{name1} ({unit1})',
                                   color_map=color_map,
                                   figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        fig = bgp.make_graph_bar2(df=df_table_no2,
                                  orientation='h',
                                  barmode='stack',
                                  width=width,
                                  height=height,
                                  opacity=0.8,
                                  reversed=True,
                                  unit=unit1,
                                  title=f"{fig_no}_{fig_start_no} {col_group} {name1} 누적 추이 (stack)",
                                  title_font_size=title_font_size,
                                  yaxes_title=f'{name1} ({unit1})',
                                  color_map=color_map,
                                  figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)

        # ------------------
        # Total :
        # ------------------

        fig = bgp.make_pie(values=s_sum1,
                           labels=s_sum1.index,
                           text1=f"{name1}<br>(전체)",
                           unit=unit1,
                           # pull_index=[6],
                           # pull_index_ratio=0.1,
                           # pull_index_ratio_default=0.01,
                           title=f"{fig_no}-{fig_start_no} {col_group} {name1} (전체)",
                           title_font_size=title_font_size,
                           colormap_name=color_map,  # "tab20c",
                           figsize=figsize)
        fig_start_no += 1
        fig_list.append(fig)
        if (year1 in df_table_no2.columns) and (year2 in df_table_no2.columns):
            fig = bgp.make_pie(values=s_sum2,
                           labels=s_sum2.index,
                           text1=f"{name1}<br>({year1}-{year2})",
                           unit=unit1,
                           # pull_index=[6],
                           # pull_index_ratio=0.1,
                           # pull_index_ratio_default=0.01,
                           title=f"{fig_no}-{fig_start_no} {col_group} {name1} ({year1}~{year2})",
                           title_font_size=title_font_size,
                           colormap_name=color_map,  # "tab20c",
                           figsize=figsize)
            fig_start_no += 1
            fig_list.append(fig)

        if year1 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year1],
                            labels=df_table_no2.index,
                            text1=f"{name1}<br>({year1})",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-{fig_start_no} {col_group} {name1}({year1})",
                            title_font_size=title_font_size,
                            colormap_name=color_map,  # "tab20c",
                            figsize=figsize)
            fig_start_no += 1
            fig_list.append(fig)

        if year2 in df_table_no2.columns:
            fig = bgp.make_pie(values=df_table_no2[year2],
                            labels=df_table_no2.index,
                            text1=f"{name1}<br>({year2})",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}-{fig_start_no} {col_group} {name1}({year2})",
                            title_font_size=title_font_size,
                            colormap_name=color_map,  # "tab20c",
                            figsize=figsize)
            fig_start_no += 1
            fig_list.append(fig)

        #fig_list = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12, fig13]

        if output == 'screen':
            for fig in fig_list:
                fig.show()
        elif output == 'file':
            for i, fig in enumerate(fig_list, start=1):
                if fig is not None: # fig None 일 때 처리 2023.6.20
                    fn = filename_prefix + f'_{i:02d}.pdf'
                    fig.write_image(fn)
                    #print(f"{fn} was saved")

        return fig_list


    def plot_3(self,
               col_group='연구비_등급4',
               col_fund='정부연구비_조',
               output='screen',
               filename=''):

        df_raw = self.df
        s_111 = df_raw[col_group].value_counts().sort_index()
        s_112 = df_raw.groupby(col_group)[col_fund].agg('sum')
        df = DataFrame([s_111, s_112])
        df.index = ['과제수 비중 ', '연구비 비중 ']
        fig1 = bgp.make_graph_bar2(df=df,
                               orientation='h',
                               yaxes_title='비율(%)',
                               xaxes_title="연구비 분포",
                               width=0.8,
                               unit='%',
                               barmode='stack',
                               precision=1,
                               to_percent=True,
                               title="연구비 규모별 과제수와 연구비 비중"
                               )

        s_r1 = df_raw.groupby(col_group)[col_fund].agg('sum')
        fig2 = bgp.make_graph_bar2(df=DataFrame(s_r1), orientation='v',
                               yaxes_title='연구비(조원)',
                               xaxes_title="연구비 분포",
                               width=0.8,
                               precision=2,
                               unit='조원'
                               )

        if output == 'screen':
            fig1.show()
            fig2.show()

        elif output == 'file':
            fn1 =  filename + '_1.pdf'
            fn2 =  filename + '_2.pdf'
            fig1.write_image(fn1)
            fig2.write_image(fn2)
            print(f"{fn1}, {fn2},files were saved.")



    def plot_histogram(self,
                         critical_number_list=[10,20,100],
                         colname_data='정부연구비_억',
                         colname_group='제출년도',
                         output='screen',
                         filename='',
                         title='',
                         title_font_size=20,
                         title2_font_size=10,
                         #  color_map='Set3',
                         figsize=(800,600)):
        """
        :param critical_number: 정부연구비 기준
        :return:
        """
        fig_list = []

        # critical_number = 10  # 100억원 이상
        df_raw = self.df


        fig = bgp.make_histogram2(df=df_raw,
                                  colname_data=colname_data,
                                  colname_group=colname_group,
                                  nbinsx=100,
                                  legend_size=8,
                                  # legend_title_size=5
                                  xaxes_title=colname_data,
                                  barmode='stack',
                                  title=f"{title} Histogram 1 (전체)",
                                  title_sub='',
                                  title_font_size=title_font_size,
                                  # color_map=color_map,
                                  figsize=figsize
                                  )
        fig_list.append(fig)

        for no_i, critical_number in enumerate(critical_number_list, start=2):
            df_raw_below = df_raw[df_raw[colname_data] <= critical_number]
            text1 = f"(0 < 연구비 <= {critical_number} 억원)"
            text1a = f'<span style="font-size: {title2_font_size}px;">{text1}</span>'
            fig = bgp.make_histogram2(df=df_raw_below,
                               colname_data=colname_data,
                               colname_group=colname_group,
                               nbinsx=100,
                               legend_size=8,
                               # legend_title_size=5
                               xaxes_title=colname_data,
                               barmode='stack',
                               title=f"{title} Histogram {no_i} ({critical_number} 억원 이하)",
                               title_sub=text1a,
                               title_font_size=title_font_size,
                               # color_map=color_map,
                               figsize=figsize
                               )
            fig_list.append(fig)

        '''
        df_raw_20_above = df_raw[df_raw[colname_data] > critical_number]
        #text2 = "(0 < 범위 < {critical_number} 억원)"
        text2 = f"({critical_number}<=  연구비 <= {df_raw_20_above[colname_data].max():.1f} 억원)"
        text2a = f'<span style="font-size: {title2_font_size}px;">{text2}</span>'
        fig = bgp.make_histogram2(df=df_raw_20_above,
                               colname_data=colname_data,
                               colname_group=colname_group,
                               nbinsx=100,
                               legend_size=8,
                               # legend_title_size=5
                               xaxes_title=colname_data,
                               barmode='stack',
                               title=f"{title} Histogram 3 ({critical_number} 이상",
                               title_sub=text2a,
                              #  color_map=color_map,
                               figsize=figsize
                               )
        fig_list.append(fig)
        '''

        if output == 'screen':
            for fig in fig_list:
                fig.show()
        elif output == 'file':
            for i, fig in enumerate(fig_list, start=1):
                fn = filename + f'_{i:02d}.pdf'
                fig.write_image(fn)


    def plot_5_horizontal_bar(self,
                              col_group='사업_부처명2',
                              col_fund='정부연구비_억',
                              PY='제출년도',
                              year1=2018, year2=2020,
                              width=0.3,
                              height=1000,
                              title1='',
                              title2='',
                              output='screen',
                              filename='',

                              fig_no=1,
                              color_map='Set3',
                              figsize=(800,600)):
        """
        가로 막대 그리기
        """
        print(">>> plot_5_horizontal_bar()")

        df_raw = self.df
        df_table = df_raw.groupby([col_group, PY])[col_fund].agg('sum').unstack(fill_value=0)
        df_table_no1 = df_table.loc[:, year1:year2]
        df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]

        fig1 = bgp.make_graph_bar2(df=df_table_no2,
                               orientation='h',
                               barmode='group',
                               width=width,
                               height=height,
                               opacity=0.8,
                               reversed=True,
                               unit=unit1,
                               title=f"{fig_no}_1 {title1}(집행액)",
                               yaxes_title='집행액(억원)',
                               color_map=color_map,
                               figsize=figsize)

        fig2 = bgp.make_pie(values=df_table_no2[year1],
                            labels=df_table_no2.index,
                            text1=f"{year1}",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title= f"{fig_no}_2 {title2}(집행액)({year1})",
                            colormap_name=color_map, # "tab20c",
                            figsize=figsize)

        fig3 = bgp.make_pie(values=df_table_no2[year2],
                            labels=df_table_no2.index,
                            text1=f"{year2}",
                            unit=unit1,
                            # pull_index=[6],
                            # pull_index_ratio=0.1,
                            # pull_index_ratio_default=0.01,
                            title=f"{fig_no}_3 {title2}(집행액)({year2})",
                            colormap_name=color_map, # "tab20c",
                            figsize=figsize)

        df_table = df_raw.groupby([col_group, PY])[col_fund].agg('count').unstack(fill_value=0)
        df_table_no1 = df_table.loc[:, year1:year2]
        df_table_no2 = df_table_no1[df_table_no1.sum(axis=1) > 0]

        fig4 = bgp.make_graph_bar2(df=df_table_no2,
                               orientation='h',
                               barmode='group',
                               width=width,
                               height=height,
                               opacity=0.8,
                               reversed=True,
                               unit=unit2,
                               title=f"{fig_no}_4 {title1}(과제수)",
                               yaxes_title='과제수',
                               color_map=color_map,
                               figsize=figsize)

        fig5 = bgp.make_pie(values=df_table_no2[year1],
                        labels=df_table_no2.index,
                        text1=f"{year1}",
                        unit=unit2,
                        # pull_index=[6],
                        # pull_index_ratio=0.1,
                        # pull_index_ratio_default=0.01,
                        title= f"{fig_no}_5 {title2}(과제수)({year1})",
                        colormap_name=color_map, # "tab20c",
                        figsize=figsize)

        fig6 = bgp.make_pie(values=df_table_no2[year2],
                        labels=df_table_no2.index,
                        text1=f"{year2}",
                        unit=unit2,
                        # pull_index=[6],
                        # pull_index_ratio=0.1,
                        # pull_index_ratio_default=0.01,
                        title=f"{fig_no}_6 {title2}(과제수)({year2})",
                        colormap_name=color_map, #"tab20c",
                        figsize=figsize)

        if output == 'screen':
            fig1.show()
            fig2.show()
            fig3.show()
            fig4.show()
            fig5.show()
            fig6.show()
        elif output == 'file':
            for i, fig in enumerate([fig1, fig2, fig3, fig4, fig5, fig6], start=1):
                fn = filename + f'_{i}.pdf'
                fig.write_image(fn)
                print(f"{fn} was saved")


    def plot_box_plot(self,
                      colname_group='제출년도',
                      colname_data='정부연구비_억',
                      figsize=(800,600),
                      title='',
                      output='file',
                      filename=''):

        df_raw = self.df
        fig_list = []
        fig = bgp.make_box_plot3(df_raw,
                                 col1=colname_group,
                                 col2=colname_data,
                                 unit=1,
                                 title=f"{title} Box Plot {colname_group} vs {colname_data}",
                                 color_map='jet', figsize=figsize)
        fig_list.append(fig)

        if output == 'screen':
            for fig in fig_list:
                fig.show()
        elif output == 'file':
            for i, fig in enumerate(fig_list, start=1):
                fn = filename + f'_{i:02d}.pdf'
                fig.write_image(fn)

    def plot_null_data(self, filename='ntis_null', PY='제출년도', TF_merger=False, TF_fig=True, fig_no=1, precision=3)->DataFrame:
        figsize = (1000, 1800)
        if self.df is not None:
            df_raw = self.df.copy()
        else:
            return None

        s_null = (~df_raw.isnull()).sum() / len(df_raw.index) * 100
        #s_null
        # s_null2 = s_null[s_null>0]
        # s_null2
        df_temp = DataFrame(s_null)
        df_temp.index = [f"{i:3d} {each}" for i, each in enumerate(df_temp.index, start=1)]
        df_temp2 = df_temp.sort_index(ascending=False)

        s_null = (df_raw.isnull()).sum() / len(df_raw.index) * 100
        # s_null
        # s_null2 = s_null[s_null>0]
        # s_null2
        df_temp = DataFrame(s_null)
        df_temp.index = [f"{i:3d} {each}" for i, each in enumerate(df_temp.index, start=1)]
        df_temp3 = df_temp.sort_index(ascending=False)
        # null 이 0인 것은 제외하기
        df_temp4 = df_temp3[~(df_temp3.loc[:,0]==0)]

        # df_temp2
        size = 15
        if TF_fig == True:
            fig = bgp.make_graph_bar2(df_temp2,
                                  width=0.5,
                                  title=f'{fig_no}-1 데이터 비율(전체)',
                                  title_sub=f'<span style="font-size: {size}px;">{df_raw.shape[0]:,}개, {len(s_null.index)}개 항목</span>',
                                  precision=precision,
                                  orientation='h',
                                  tickfont_size=5,
                                  unit=" % ",
                                  figsize=figsize,
                                  # yaxes_title=f'널 비율(%) '
                                  )
            fig.write_image(f'{filename}_1.pdf')

            fig = bgp.make_graph_bar2(df_temp4,
                                      width=0.5,
                                      title=f'{fig_no}-2 널 존재하는 컬럼 추출 - 데이터 비율(전체)',
                                      title_sub=f'<span style="font-size: {size}px;">{df_raw.shape[0]:,}개, {len(df_temp4.index)}개 항목</span>',
                                      precision=precision,
                                      orientation='h',
                                      tickfont_size=5,
                                      unit=" % ",
                                      figsize=figsize,
                                      # yaxes_title=f'널 비율(%) '
                                      )
            fig.write_image(f'{filename}_2.pdf')


        # null data
        #PY = '제출년도'
        s1 = df_raw[PY].value_counts()
        s1a = s1.sort_index()

        Y1 = s1.index.min()
        Y2 = s1.index.max()

        df_count = df_raw.groupby([PY]).agg('count').T
        df_count2 = df_count.divide(s1a, axis=1)
        df_count3 = 100 * df_count2

        # 제출년도 포함 시켜주기 : 모두 100 으로 설정해 주기
        df_1 = DataFrame([[100] * len(df_count3.columns)],
                         columns=df_count3.columns,
                         index=[PY])
        df_count3 = pd.concat([df_1, df_count3], axis=0)

        # index 조정
        df_count4 = df_count3.copy()
        df_count4.index = [f" {i}. {each}" for i, each in enumerate(df_count4.index, start=1)]

        if TF_fig == True:
            df_fig = df_count4
            fig = bgp.make_heatmap(title=f"{fig_no}-3 데이터 비율(연도별 : {Y1}~{Y2})",
                               df=df_fig,
                               colorscale='blues',  # 'greens',
                               textfont_size=3,
                               figsize=figsize,
                               margin=dict(l=20, r=20, t=20, b=20),
                               reversed=True,
                               colorbar_len=0.1,
                               colorbar_orientation='h',
                               colorbar_ticklen=2,
                               colorbar_tickfont_size=8,
                               minimum_number=0.3,  # 이 숫자 아래는 텍스트 출력하지 않기
                               # x_min=1998,
                               # x_max=2021,
                               )
            fig.write_image(f'{filename}_3.pdf')

        if (TF_fig == True) and (TF_merger == True):
            merger = PdfMerger()
            for fn in sorted([f'{filename}_1.pdf', f'{filename}_2.pdf', f'{filename}_3.pdf']):
                merger.append(fn)
            fname2 = bu.make_new_name(f'total_{filename}', 'pdf')
            merger.write(fname2)
            merger.close()

        return df_count3





    def make_table_ptn(self, df=None,
                           column_index_list=['사업_부처명', '사업명', '내역사업명'],
                           column_search='사업명',


                           column_FUND='정부연구비(원)',
                           ptn_list=[],
                           filename=None, # 'ex_ptn1.xlsx',
                           #width_00=20,
                           only_summary= False,
                           show=False,
                           dic_column_width={'사업_부처명': 20, '사업명': 50, '사업수(개)': 10,
                                         '부처기준사업수(개)': 10, '비교': 10, '과제수(개)': 10,
                                         '패턴': 25,
                                         'memo': 30},
                          ):
        if df is None:
            if self.df is not None:
                df = self.df.copy()
            else:
                print("check your data !")
                return None
        if ptn_list :

            df_table_ptn = bnf.make_table_ptn(df=df,
                                          column_search=column_search,
                                          column_index_list=column_index_list,
                                          column_FUND=column_FUND,
                                          ptn_list=ptn_list,
                                          filename=filename,
                                          # width_00=20,
                                          only_summary=only_summary,
                                          show=show,
                                          dic_column_width=dic_column_width)
            return df_table_ptn
        else:
            print(">>> ptn_list is empty !")
            return None


    def make_pdf(self, filename_prefix='prefix_',
                 year1=None, year2=None, PY='제출년도',
                 color_map='tab20c', figsize=(800,600),
                col_fund1='정부연구비_조',
                col_fund2='정부연구비_억',
                col_groups_fund=['연구비_등급1'],  # , '연구비_등급2', '연구비_등급4'],
                col_groups=[
                          # '사업_부처명',
                            '사업_부처명m',
                          #  '연구수행주체',
                            '과제수행기관명',
                          #  '지역', '지역2', '지역3',
                          #  '6T관련기술-대', '과학기술표준분류1-대', '중점과학기술분류-대',
                          #  '연구개발단계',
                          #  '연구책임자성별',
                          #  '보안과제3',
                          #  '연구비_등급1', '연구비_등급2', '연구비_등급4'
                        ],
                 title_font_size=20,  text_cagr_font_size=10, width_string=200,
                 critical_number_list=[10,20,100], # 10억, 20억, 100억
                 range_l=None, range_r=None,
                 onoff_condition=['null', 'fund_trend1',
                                  'fund_histogram', 'fund_boxplot',
                                  'fund_trend2',
                                  'col_groups'],
                 onoff_name='A'):
        """
        2023.5.24
        2023.6.14
        """
        print(">>> make_pdf()")
        # 전체 집행액, 과제수

        if year1 is None:
            year1 = self.df[PY].min()
        if year2 is None:
            year2 = self.df[PY].max()

        fig_no_name = 'A1'

        # 작업편의를 위해서 만듦
        if onoff_name=='A':
            onoff_condition = ['null', 'fund_trend1', 'fund_histogram', 'fund_boxplot']
        elif onoff_name=='A1':
            onoff_condition = ['null']
        elif onoff_name == 'AB':
            onoff_condition = ['null', 'fund_trend1', 'fund_histogram', 'fund_boxplot', 'col_groups']
        elif onoff_name == 'B':
            onoff_condition = ['col_groups']

        # 1 널 데이터 현황
        filename1 = f'{filename_prefix}{fig_no_name}_1_null_total'
        if 'null' in onoff_condition:
            print("(1) draw null")
            df_count3 = self.plot_null_data(filename=filename1, TF_fig=True, fig_no=fig_no_name)
        else:
            df_count3 = self.plot_null_data(filename=filename1, TF_fig=False, fig_no=fig_no_name)
            #print(f"df_count3.shape={df_count3.shape}")

        # 2 연구비, 과제수 현황, 연구비 등급별 분포
        if 'fund_trend1' in onoff_condition:
            print("(2) fund_trend1()")
            fig_no_name = 'A2'

            self.plot_1(col_fund1=col_fund1, title1=f"{fig_no_name}-1 정부 연구개발 집행액과 과제수",
                        col_fund2=col_fund2, title2=f"{fig_no_name}-2 정부 연구개발 집행액의 평균 및 중앙값",
                        output='file',
                        filename=f'{filename_prefix}{fig_no_name}_2_{col_fund1}_연구비_과제수',
                        PY=PY,
                        color_map=color_map,
                        figsize=figsize,
                        title_font_size=title_font_size,
                        text_cagr_font_size=text_cagr_font_size,
                        width_string=width_string,
                        range_l=range_l, range_r=range_r,
                         )
        if 'fund_histogram' in onoff_condition:
            print('(3) fund_histogram()')
            fig_no_name = 'A3'
            #fig_no += 1
            # histogram
            self.plot_histogram(critical_number_list=critical_number_list,
                                  colname_data=col_fund2, # '정부연구비_억',
                                  colname_group=PY,
                                  output='file',
                                  filename=f'{filename_prefix}{fig_no_name}_{col_fund2}',
                                  #color_map=color_map,
                                  title=f"{fig_no_name} {col_fund2}",
                                  title_font_size=title_font_size,
                                  figsize=figsize,
                                  )

        if 'fund_boxplot' in onoff_condition:
            print('(4) fund_boxplot()')
            fig_no_name = 'A4'
            # box_plot
            self.plot_box_plot(colname_group=PY, colname_data=col_fund2,
                          figsize=figsize,
                          title=f"{fig_no_name}-{col_fund2}",
                          filename=f'{filename_prefix}{fig_no_name}_{col_fund2}',
                          output='file')


        # 그림번호 3 이상
        fig_no_name = 'B'
        if 'col_groups' in onoff_condition:
            # col_group 별 집행 현황 분석
            for each_fig_no, col_group1 in enumerate(col_groups, start=1):
                print(f"{each_fig_no} : {col_group1}")
                #fig_no += 1
                if col_group1 in self.df.columns:
                    self.plot_2(col_group=col_group1, output='file', filename=f'{filename_prefix}{fig_no_name}_{each_fig_no:03d}_{col_group1}',
                                fig_no=f"{fig_no_name}-{each_fig_no}",
                                color_map=color_map, figsize=figsize, year1=year1, year2=year2,
                                title_font_size=title_font_size, text_cagr_font_size=text_cagr_font_size, df_null_count=df_count3)
                else:
                    print(f"(warning!) {col_group1} is not exist in df.columns")

        # filename_prefix로 시작하는 파일명 연결 하기
        filenames1 = glob.glob(f'{filename_prefix}*.pdf')
        filenames = sorted(filenames1)
        merger = PdfMerger()

        for fn in sorted(filenames):
            merger.append(fn)

        fname2 = bu.make_new_name(f'total_{filename_prefix}_{color_map}', 'pdf')
        merger.write(fname2)
        merger.close()
        print(f" *** {fname2} was created ! *** ")

if __name__ == "__main__":
    print("bk_ntis.py")
