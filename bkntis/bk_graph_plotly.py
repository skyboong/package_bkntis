# bk_graph_ploty.py
# plotly로 그래프를 그리는 함수를 정리하여 모아 놓았다.
# graph_plotly.py, bk_graph_plotly_win2.py 에 있는 것을 통합하여 만듦
# 필요한 BK 모듈 파일 이름 : bk_util.py 불러옴
# update : 2023.5.3(Wed)

import sys
import os
import glob
import functools

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validators.scatter.marker import SymbolValidator
from matplotlib import cm
import matplotlib.colors as mcolor
from PyPDF2 import PdfMerger 


sys.path.append("/Users/bk/Dropbox/bkmodule2019/")

import bk_util as bu

def deco1(func):
    @functools.wraps(func)
    def wrap(*args, **kwargs):
        print( f"\n>>> {func.__name__}( )")
        return func(*args, **kwargs)
    return wrap

raw_symbols = SymbolValidator().values

def find_hex_color_from_cmap(x, colormap_name=None):
    """
    지정한 colormap에 0 ~ 1.0 사이 값을 입력하면, 16진수 형태로 컬러값을 돌려줌
    2022.4.11
    """
    if colormap_name is None:
        colormap_name = 'tab20c'

    colormap1 = cm.get_cmap(colormap_name)
    color_rgb = colormap1(x)  # ex [0.1, 0.2, 0.3]
    color_hex = mcolor.to_hex(color_rgb)
    return f"{color_hex}"


def make_table1(df=None, *, money_unit=1000000000000):
    """컬럼 입력 받아서 년도별 합계 구함(연구금액, 건수)"""

    col_1 = '제출년도'
    col_2 = '정부연구비(원)'
    df2 = df.groupby([col_1])[col_2].agg('sum')
    s2 = df2 / money_unit  # 조원

    s4 = df[col_1].value_counts().sort_index()
    df_g1 = DataFrame([s2, s4]).T

    if money_unit == 100000000:
        aa = '억원'
    elif money_unit == 1000000000:
        aa = '십억원'
    elif money_unit == 10000000000:
        aa = '백억원'
    elif money_unit == 100000000000:
        aa = '천억원'
    elif money_unit == 1000000000000:
        aa = '조원'
    else:
        aa = ''

    df_g1a = df_g1.rename(columns={col_1: "과제수", col_2: f'집행액({aa})'})
    df_g1a['과제수'] = df_g1a['과제수'].astype(int)
    return df_g1a


def make_table_no8(df,
                   col_name='연구개발단계',
                   col_1='제출년도',
                   col_2='정부연구비(원)',
                   option=1):
    """
    연구개발단계별 트렌드를 총액 기준 또는 퍼센트(%) 기준으로 리턴함

    col_1 = '제출년도'
    col_2 = '정부연구비(원)'


    option 1: 총액기준
           2: 퍼센트
    """
    df2 = df.groupby([col_name, col_1])[col_2].agg(np.sum).unstack(fill_value=0)
    df3 = df2.sort_index()
    if option == 2:
        df4 = df3 / df3.sum(axis=0)
        df5 = 100 * df4
    else:
        df5 = df3
    return df5


def make_table_no9(df,
                   col_name='연구개발단계',
                   col_1='제출년도',
                   col_2='정부연구비(원)',
                   method='sum',
                   option=1):
    """
    연구개발단계별 트렌드를 총액 기준 또는 퍼센트(%) 기준으로 리턴함

    col_1 = '제출년도'
    col_2 = '정부연구비(원)'


    option 1: 총액기준
           2: 퍼센트
    """
    if method == 'sum':
        df2 = df.groupby([col_name, col_1])[col_2].agg(np.sum).unstack(fill_value=0)
    elif method == 'count':
        df2 = df.groupby([col_name, col_1])[col_2].agg('count').unstack(fill_value=0)
    else:
        df2 = df.groupby([col_name, col_1])[col_2].agg(np.sum).unstack(fill_value=0)

    df3 = df2.sort_index()
    if option == 2:
        df4 = df3 / df3.sum(axis=0)
        df5 = 100 * df4
    else:
        df5 = df3
    return df5


def make_scatter_and_line2(df, *, col_left=None, col_right=None, range1=None, range2=None, **kwargs):
    """
    양쪽이 스케일 다른 그래프 나타내기
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    unit = df.columns[0].split('(')
    unit2 = unit[1].split(')')[0]

    text1 = [f"{txt:.2f}{unit2}" for txt in df[col_left].tolist()]
    text2 = [f"{int(txt):,}개" for txt in df[col_right].tolist()]
    # print(text2)

    # Add traces
    fig.add_trace(
        go.Bar(x=df.index, y=df[col_left],
               width=0.5,
               marker={'color': "#F7DC6F",
                       'line': {'color': '#ffffff', 'width': 3}},
               text=text1,
               hovertemplate='%{x}, %{y}',
               # textposition='auto', ['inside', 'outside', 'auto', 'none']
               # textposition='inside',
               textposition='outside',
               textfont={'color': '#222222', 'size': 10},
               name="집행액"),
        secondary_y=False,
        # secondary_y=True,
    )
    # textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)

    fig.add_trace(
        go.Scatter(x=df.index, y=df[col_right],
                   marker={'color': '#633974',
                           'size': 16,
                           'line': {'color': '#ffffff', 'width': 3}},
                   text=text2,
                   hovertemplate='%{x}, %{y}',
                   textfont={'color': '#222222', 'size': 10},
                   textposition='top right',
                   mode='lines+markers+text',
                   name="과제수"),
        secondary_y=True,
        # secondary_y=False,
    )

    # fig.update_traces(bgcolor='green')

    # Add figure title
    fig.update_layout(
        # title_text="국가연구개발사업 집행액과 세부과제수(2016-2020년)",
        title={'text': kwargs.get('title_pre', '') + kwargs.get('title', '국가연구개발사업 집행액과 세부과제수(2016-2020년)'),
               'font': {'size': 16},
               'xanchor': 'left'},
        plot_bgcolor='#eeeeee',  # 차트 안의 색
    )

    # Set x-axis title
    fig.update_xaxes(title_text="연도")

    # Set y-axes titles

    if range1 is None:
        range1 = [0, df[col_left].max() * 1.2]
        # print(range1)
    if range2 is None:
        range2 = [0, df[col_right].max() * 1.2]
        # print(range2)
    fig.update_yaxes(title_text=f"<b>집행액</b>({unit2})", secondary_y=False, range=range1)
    fig.update_yaxes(title_text="<b>세부과제수</b>(개)", secondary_y=True, range=range2, showgrid=False)

    fig.show()

def make_scatter_and_line3(df, *,
                           col_left=None,
                           col_right=None,
                           range1=None,
                           range2=None,
                           unit1='',
                           unit2='',
                           title_xaxes='연도',
                           title_yaxes_left='',
                           title_yaxes_right='',
                           name1='집행액',
                           name2='과제수',
                           precision1=1,
                           precision2=2,
                           marker_size=16,
                           marker_width=3,
                           textfont_size1=10,
                           textfont_size2=10,
                           textposition1='outside',
                           textposition2='top right',
                           title_pre='',
                           title='',
                           title_font_size=20,
                           cagr=False,
                           n_high=5,
                           text_cagr_font_size=5,
                           width_string=150,
                           figsize=(800,600),
                           plot_bgcolor='rgba(230,236,245)',
                           bar_marker_color="#F7DC6F",
                           bar_marker_line_width=3,
                           bar_marker_line_color="#ffffff",
                           xaxes_ticktext=None,
                           xaxes_tickvals=None,
                           ):
    """양쪽이 스케일 다른 그래프 나타내기

    """
    print(">>>")
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    text1 = [f"{txt:,.{precision1}f}{unit1}" for txt in df[col_left].tolist()]
    text2 = [f"{txt:,.{precision2}f}{unit2}" for txt in df[col_right].tolist()]

    # CAGR 표시
    '''
    if cagr:
        df2 = df[[col_left, col_right]].T
        # print(df2)
        pearson1 = bu.make_pearson_v2(df=df2, delta=0)
        text = [f"{each['word']}: {each['cagr']:.1f}%" for index, each in pearson1.iterrows()]
        text_cagr = ' CAGR = ' + ', '.join(text) + f" ({df2.columns[0]}~{df2.columns[-1]})"
    else:
        text_cagr = ''
    '''


    # CAGR 표시
    if cagr:
        # 2022.9.14
        df2 = df[[col_left, col_right]].T
        pearson1 = bu.make_pearson_v3(df=df2, delta=0)
        pearson2 = pearson1.sort_values(by=['cagr'], ascending=[False])
        text = [f"{each['word']}: {each['cagr']:.1f}%" for index, each in pearson2.iterrows()]
        if n_high < len(text):
            pass
        else:
            n_high = len(text)
        text_cagr = ' CAGR = ' + ', '.join(text[:n_high])
        text_cagr = f'<span style="font-size: {text_cagr_font_size}px;">{text_cagr}</span>'
        text_cagr = bu.make_fixedwidth_string(txt=text_cagr, width=width_string, sep='<br>')
    else:
        text_cagr = ''



    # Add traces
    fig.add_trace(
        go.Bar(x=df.index, y=df[col_left],
               width=0.5,
               marker={'color': bar_marker_color,
                       'line': {'color': bar_marker_line_color, 'width': bar_marker_line_width}},
               text=text1,
               hovertemplate='%{x}, %{y}',
               # textposition='auto', ['inside', 'outside', 'auto', 'none']
               # textposition='inside',
               textposition=textposition1,
               textfont={'color': '#222222', 'size': textfont_size1},
               name=name1),
        secondary_y=False,
        # secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df[col_right],
                   marker={'color': '#633974',
                           'size': marker_size,
                           'line': {'color': '#ffffff', 'width': marker_width}},
                   text=text2,
                   hovertemplate='%{x}, %{y}',
                   textfont={'color': '#222222', 'size': textfont_size2},
                   textposition=textposition2,  # 'top right',
                   mode='lines+markers+text',
                   name=name2),
        secondary_y=True,
        # secondary_y=False,
    )

    # fig.update_traces(bgcolor='green')

    # Add figure title
    fig.update_layout(
        # title_text="국가연구개발사업 집행액과 세부과제수(2016-2020년)",
        #'text': title + '<br>' + text_cagr,
        title={'text': title_pre + title + '<br>' + text_cagr,
               'font': {'size': title_font_size},
               'xanchor': 'left'},
        plot_bgcolor=plot_bgcolor,  # 차트 안의 색
        width=figsize[0],
        height=figsize[1]
    )

    if range1 is None:
        range1 = [0, df[col_left].max() * 1.2]
        # print(range1)
    if range2 is None:
        range2 = [0, df[col_right].max() * 1.2]
        # print(range2)
    fig.update_yaxes(title_text=f"{title_yaxes_left}{unit1}", secondary_y=False, range=range1)
    fig.update_yaxes(title_text=f"{title_yaxes_right}{unit2}", secondary_y=True, range=range2, showgrid=False)

    # legend position
    fig.update_layout(legend=dict(yanchor="bottom", y=-0.2, xanchor="center", x=0.5, orientation="h"))
    if xaxes_ticktext is not None:
        print(f"xaxes_ticktext={xaxes_ticktext}")
        fig.update_xaxes(ticktext=xaxes_ticktext, tickvals=xaxes_tickvals)
    # fig.show()
    print(f"* {title}")
    return fig


def make_scatter(df,
                 col_x=None,
                 col_y=None,
                 col_size=None,
                 range_x=None,
                 range_y=None,
                 unit_x='',
                 unit_y='',
                 title_xaxes='연도',
                 title_yaxes='',
                 name='집행액',
                 marker_size=1,
                 precision=1,
                 **kwargs):
    fig = go.Figure()

    if col_size is None:
        df['col_size'] = 1
        col_size = 'col_size'

    # Add traces
    fig.add_trace(
        go.Scatter(x=df[col_x],
                   y=df[col_y],
                   marker={'color': '#633974',
                           'size': marker_size,  # df[col_size],
                           'line': {'color': '#ffffff', 'width': 3}},
                   # text=df[col_size],
                   hovertemplate='%{x}, %{y}',
                   textfont={'color': '#222222', 'size': 10},
                   textposition='top right',
                   mode='markers',
                   name=name),
    )

    # Add figure title
    fig.update_layout(
        # title_text="국가연구개발사업 집행액과 세부과제수(2016-2020년)",
        title={'text': kwargs.get('title_pre', '') + kwargs.get('title', ''),
               'font': {'size': 16},
               'xanchor': 'left'},
        plot_bgcolor='#eeeeee',  # 차트 안의 색
    )

    if range_x is None:
        range_x = [0, df[col_x].max() * 1.2]
        # print(range1)
    if range_y is None:
        range_y = [0, df[col_y].max() * 1.2]
        # print(range2)
    fig.update_xaxes(title_text=f"{title_xaxes}{unit_x}", range=range_x)
    fig.update_yaxes(title_text=f"{title_yaxes}{unit_y}", range=range_y, showgrid=False)

    fig.show()


def make_scatter2(df,
                  col_x=None,
                  col_y=None,
                  col_name=None,
                  col_size=None,
                  col_size_alpha=10,
                  col_color=None,
                  range_x=None,
                  range_y=None,
                  unit_x='',
                  unit_y='',
                  title_xaxes=None,
                  title_yaxes=None,
                  name='집행액',
                  mode='markers',
                  marker_size=1,
                  marker_line_width=1,
                  precision=1,
                  textposition='top right',
                  dic_textfont={'color': '#222222', 'size': 10},
                  figsize=(800, 800),
                  hline=False,
                  vline=False,
                  annotation=True,
                  annotation_xy=[0, 0],
                  annotation_text='',
                  margin={},
                  **kwargs):
    print(f">> make_scatter2(figsize={figsize})")

    fig = go.Figure()

    if col_size is None:
        col_size = 1
    elif isinstance(col_size, int):
        col_size = col_size
    else:
        col_size = (col_size_alpha * df[col_size] / df[col_size].max()).tolist()
        # print(col_size)

    if col_color is not None:
        col_color = df[col_color].tolist()
    else:
        col_color = col_size

    # col_name 이 유효하면 출력하고 그렇지 않으면 출력하지 않음
    if col_name in df.columns:
        text2 = [f"{each1}" for each1 in df[col_name].tolist()]
    else:
        text2 = ["" for each1 in range(len(df.index))]

    # print(text2)
    # Add traces
    fig.add_trace(

        go.Scatter(x=df[col_x],
                   y=df[col_y],
                   text=text2,
                   marker={'color': col_color,  # '#633974',
                           'size': col_size,  # marker_size, # df[col_size],
                           'line': {'color': '#ffffff', 'width': marker_line_width}},
                   # text=df[col_size],
                   hovertemplate='%{text} (%{x}, %{y})',
                   textfont=dic_textfont,  # {'color': '#222222', 'size': 10},
                   textposition=textposition,  # 'top right',
                   mode=mode,
                   name=name),
        # Any    combination    of['lines', 'markers', 'text']     joined    with '+' characters    (e.g.'lines+markers')
    )

    # Add figure title
    fig.update_layout(
        autosize=False,
        width=figsize[0],
        height=figsize[1],
        # title_text="국가연구개발사업 집행액과 세부과제수(2016-2020년)",
        title={'text': kwargs.get('title_pre', '') + kwargs.get('title', ''),
               'font': {'size': 16},
               'xanchor': 'left'},
        plot_bgcolor='#eeeeee',  # 차트 안의 색
        margin=margin
    )
    # 2022.9.15
    if hline == True:
        fig.add_hline(y=df[col_y].mean(), line_width=1, line_dash="dot", line_color="#ff0000")
    if vline == True:
        fig.add_vline(x=df[col_x].mean(), line_width=1, line_dash="dot", line_color="#ff0000")

    if range_x is None:
        range_x = [0, df[col_x].max() * 1.2]
        # print(range1)
    if range_y is None:
        range_y = [0, df[col_y].max() * 1.2]
        # print(range2)

    if title_xaxes is None:
        title_xaxes = col_x
    if title_yaxes is None:
        title_yaxes = col_y
    fig.update_xaxes(title_text=f"{title_xaxes}{unit_x}", range=range_x)
    fig.update_yaxes(title_text=f"{title_yaxes}{unit_y}", range=range_y, showgrid=False)

    # fig.show()
    if annotation == True:
        fig.add_annotation(x=annotation_xy[0], y=annotation_xy[1],
                           text=annotation_text,
                           align="left",  # "right",
                           showarrow=False,
                           textangle=0,
                           xanchor='left',
                           xref="paper",
                           yref="paper")

    return fig


def make_graph_pie(df=None,
                   col_value='',
                   col_label='',
                   index=False,
                   hole1=.3,
                   title='',
                   text='',
                   colors='#ff0000',
                   colormap='tab20c',
                   minsize=1,
                   font_size=20,
                   textposition='inside',
                   figsize=(800,600)):
    """
    파이 그래프 그리기,
    DataFrame 입력받아서 label, value 컬럼 선택

    2022.4.13(Wed)
    2022.4.19(Tue)
    """

    if len(df.index) < 1:
        print("length of df is less than 1")
        return False

    # label 설정 : index 가 True 이면, label 을 index 값 지정
    if index:
        labels = df.index
    else:
        labels = df[col_label]

    # value 설정 : 데이타프레임 또는 시리즈
    if isinstance(df, DataFrame):
        values = df[col_value]
    else:
        s1 = df
        values = s1.tolist()

    col_number = len(df.index)

    colors = []
    for i in range(col_number):
        c1 = find_hex_color_from_cmap(i / col_number, colormap)
        colors.append(c1)

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=hole1)])
    fig.update_traces(
        textinfo='percent+label+value',
        texttemplate='%{label}<br>%{value:,.1f}<br>(%{percent})',
        textposition=textposition,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=1))
    )
    fig.update_layout(
        title_text=title,
        # Add annotations in the center of the donut pies.
        annotations=[dict(text=text, x=0.5, y=0.5, font_size=font_size, showarrow=False)],
        uniformtext_minsize=minsize,
        uniformtext_mode='hide',
        width=figsize[0],
        height=figsize[1],
    )
    fig.show()
    # return fig


def make_graph_pie_single(values=None,
                          labels=None,
                          hole1=.3,
                          title='',
                          text='',
                          colors='#ff0000',
                          colormap='jet',  # 'tab20c',
                          minsize=1,
                          font_size=20,
                          font_size_label=15,
                          textposition1='inside',
                          textposition2='outside',
                          pull=None,
                          figsize=(800, 600),
                          showlegend=False,
                          annotation=True,
                          annotation_xy=[0, 0],
                          annotation_text='',
                          margin={}
                          ):
    print(f">>> make_graph_pie_single(figsize={figsize})")

    col_number = len(values)
    colors_bg = []
    colors_txt = []
    for i in range(col_number):
        c1 = find_hex_color_from_cmap(i / col_number, colormap)
        colors_bg.append(c1)
        colors_txt.append(c1)

    if pull is None:
        pull = [0] * len(labels)

    font_size_list1 = []
    for i in range(len(values)):
        x = font_size_label - 2 * i
        min_size_label = minsize
        if x > min_size_label:
            font_size_list1.append(x)
        else:
            font_size_list1.append(min_size_label)

    # print(f"font_size_list1={font_size_list1}")
    common_props = dict(labels=labels,
                        values=values,
                        hole=hole1,
                        pull=pull)

    trace1 = go.Pie(
        **common_props,
        textinfo='percent+value',
        # texttemplate='%{value:,.1f}<br>(%{percent})',
        texttemplate='%{value:,.0f}<br>(%{percent})',
        textposition='outside')

    trace2 = go.Pie(
        **common_props,
        textinfo='label',
        textposition='inside',
        # textfont = {'family': "Times", 'size': [15]*7, 'color':['#ffffff']*7 },
        textfont={'size': font_size_list1, 'color': ['#ffffff']},
        insidetextorientation='auto'
    )

    fig = go.Figure(data=[trace1, trace2])
    fig.update_layout(
        title_text=title,
        # Add annotations in the center of the donut pies.
        # 도너 중앙에 들어갈 글
        annotations=[dict(text=text, x=0.5, y=0.5, font_size=font_size, showarrow=False)],
        # uniformtext_minsize=minsize,
        # uniformtext_mode='hide'
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        margin=margin,
    )
    if annotation == True:
        fig.add_annotation(x=annotation_xy[0] - 0.15, y=annotation_xy[1],
                           text=annotation_text,
                           align="left",  # "right",
                           showarrow=False,
                           textangle=0,
                           xanchor='left',
                           xref="paper",
                           yref="paper")
    # fig.show()
    return fig


def make_graph_pie_single_with_left_value(values=None,
                                          labels=None,
                                          hole1=.3,
                                          title='',
                                          text='',
                                          colors='#ff0000',
                                          colormap='jet',  # 'tab20c',
                                          minsize=1,
                                          font_size=20,
                                          font_size_label=15,
                                          textposition1='inside',
                                          textposition2='outside',
                                          pull=None,
                                          figsize=(400, 400),
                                          showlegend=False,
                                          annotation=True,
                                          annotation_xy=[0, 0],
                                          annotation_text='',
                                          margin={},
                                          left_value=[10, 20],
                                          left_label=['OUT', 'IN'],
                                          left_textposition='auto',
                                          ):
    print(f">>> make_graph_pie_single_with_left_value(figsize={figsize})")
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'xy'}, {'type': 'domain'}]])

    # Top left
    left_x = ['total'] * len(left_value)
    left_marker_color = ['#7f7f7f'] * (len(left_value) - 1) + ['#d62728']  #
    left_width = [0.1] * len(left_value)
    trace0 = go.Bar(x=left_x,
                    y=left_value,
                    name="IN_OUT",
                    # text=left_label,
                    text=[f"{each_l}<br>{each_v:,.1f}({100 * each_v / sum(left_value):.1f}%)" \
                          for each_l, each_v in zip(left_label, left_value)],
                    marker_color=left_marker_color,
                    width=left_width,
                    textposition=left_textposition)

    col_number = len(values)
    colors_bg = []
    colors_txt = []
    for i in range(col_number):
        c1 = find_hex_color_from_cmap(i / col_number, colormap)
        colors_bg.append(c1)
        colors_txt.append(c1)

    if pull is None:
        pull = [0] * len(labels)

    font_size_list1 = []
    for i in range(len(values)):
        x = font_size_label - 2 * i
        min_size_label = 7
        if x > min_size_label:
            font_size_list1.append(x)
        else:
            font_size_list1.append(min_size_label)

    # print(f"font_size_list1={font_size_list1}")

    common_props = dict(labels=labels,
                        values=values,
                        hole=hole1,
                        pull=pull)
    # 바깥쪽 지정
    trace1a = go.Pie(
        **common_props,
        textposition='outside',
        textinfo='percent+value',
        texttemplate='%{value:,.1f}(%{percent})',
    )
    # 안쪽 지정
    trace1b = go.Pie(
        **common_props,
        textposition='inside',
        textinfo='label',
        # textfont = {'family': "Times", 'size': [15]*7, 'color':['#ffffff']*7 },
        textfont={'size': font_size_list1, 'color': ['#ffffff']},
        insidetextorientation='auto',
    )

    fig.add_trace(trace0, row=1, col=1)
    fig.add_trace(trace1a, row=1, col=2)
    fig.add_trace(trace1b, row=1, col=2)
    # fig.add_data=[trace1, trace2])
    # fig = go.Figure(data=[trace1])
    fig.update_layout(
        title_text=title,
        # 도너 중앙에 들어갈 글
        annotations=[dict(text=text, x=0.5, y=0.5, font_size=font_size, showarrow=False)],
        # uniformtext_minsize=minsize,
        # uniformtext_mode='hide'
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        margin=margin,
        barmode='stack',
    )
    if annotation == True:
        fig.add_annotation(x=annotation_xy[0], y=annotation_xy[1],
                           text=annotation_text,
                           align="left",  # "right",
                           showarrow=False,
                           textangle=0,
                           xanchor='left',
                           xref="paper",
                           yref="paper")
    # fig.show()
    return fig


def make_graph_pie_multi(df=None,
                         hole1=.3,
                         title='',
                         text='',
                         colors='',
                         colormap_name=None,
                         minsize=1,
                         pos_x_list=[],
                         textposition='inside',
                         sort=True,
                         font_size=10,
                         annotation_align='center',
                         precision=0,
                         unit='',
                         showlegend=False):
    """
    다수의 파이 그래프 그리기,
    조건. 인덱스가 레이블이 됨

    2022.4.13(Wed)
    2022.4.19(Tue)
    """

    if len(df.index) < 1:
        print("length of df is less than 1")
        return False

    labels = df.index

    col_number = len(df.columns)

    # 색 설정
    if colors == '':
        colors2 = []
        for i in range(col_number):
            c1 = find_hex_color_from_cmap(i / col_number, colormap_name=colormap_name)
            colors2.append(c1)
        colors = colors2

    if precision == 0:
        texttemplate1 = '%{label}<br>%{value:,.0f}<br>(%{percent})'
    elif precision == 1:
        texttemplate1 = '%{label}<br>%{value:,.1f}<br>(%{percent})'
    elif precision == 2:
        texttemplate1 = '%{label}<br>%{value:,.2f}<br>(%{percent})'
    else:
        texttemplate1 = '%{label}<br>%{value:,.3f}<br>(%{percent})'

    if precision == 0:
        texttemplate1 = '%{label}<br>%{value:,.0f}'
    elif precision == 1:
        texttemplate1 = '%{label}<br>%{value:,.1f}'
    elif precision == 2:
        texttemplate1 = '%{label}<br>%{value:,.2f}'
    else:
        texttemplate1 = '%{label}<br>%{value:,.3f}'
    texttemplate1 = texttemplate1 + f"{unit}" + '<br>(%{percent})'

    specs_list = []
    for each in range(col_number):
        specs_list.append({'type': 'domain'})
        # ’domain’: Subplot type for traces that are individuall positioned. pie, parcoords, parcats, etc.
    fig = make_subplots(rows=1, cols=col_number, specs=[specs_list])

    j = 0
    for each_col in df.columns:
        j += 1
        fig.add_trace(go.Pie(labels=labels, values=df[each_col]), 1, j)

    fig.update_traces(
        hole=hole1,
        textinfo='percent+label+value',
        texttemplate=texttemplate1,
        textposition=textposition,
        marker=dict(colors=colors, line=dict(color='#ffffff', width=1)),
        sort=sort
    )

    annotation_list = []

    # annotation

    # print("delta = ", delta)
    print("x_list = ", pos_x_list)
    for xi, each_col in zip(pos_x_list, df.columns):
        temp = dict(text=f"{each_col}", x=xi, y=0.5, font_size=font_size, showarrow=False, align=annotation_align)
        annotation_list.append(temp)

    # print("align=", annotation_align)
    fig.update_layout(
        title_text=title,
        # Add annotations in the center of the donut pies.
        annotations=annotation_list,
        uniformtext_minsize=minsize,
        uniformtext_mode='hide',
        showlegend=showlegend
    )
    fig.show()
    # return fig


def make_pie(*,
             values=[],
             labels=[],
             hole1=.3,
             colors='',
             pull_index=[],
             pull_index_ratio=0.1,
             pull_index_ratio_default=0.,  # 0-1.0
             textposition='inside',
             colormap_name=None,
             precision=0,
             unit='',
             title='',
             title_font_size=20,
             text1='',
             text1_font_size=20,
             sort=True,
             figsize=(800,600)):
    # **kwargs):
    """
    파이 그래프
    2022.4.13(Wed)
    """

    # 색 설정
    col_number = len(values)
    if colors == '':
        colors2 = []
        for i in range(col_number):
            c1 = find_hex_color_from_cmap(i / col_number, colormap_name=colormap_name)
            colors2.append(c1)
        colors = colors2

    # pull 설정하게 해줌, 여러개 입력받아서, 동일한 비율로 부각되게 함
    pull = [pull_index_ratio_default for i in range(len(values))]
    pull_index_list = []
    if isinstance(pull_index, list):
        for each_index in pull_index:
            if each_index < len(values):
                pull_index_list.append(each_index)
    for index, each in enumerate(pull):
        if index in pull_index_list:
            pull[index] += pull_index_ratio

    if len(values) != len(labels):
        return False

    fig = go.Figure(data=[go.Pie(labels=labels,
                                 values=values,
                                 hole=hole1,
                                 pull=pull,
                                 sort=sort)])
    if precision == 0:
        texttemplate1 = '%{label}<br>%{value:,.0f}'
    elif precision == 1:
        texttemplate1 = '%{label}<br>%{value:,.1f}'
    elif precision == 2:
        texttemplate1 = '%{label}<br>%{value:,.2f}'
    else:
        texttemplate1 = '%{label}<br>%{value:,.3f}'
    texttemplate1 = texttemplate1 + f"{unit}" + '<br>(%{percent})'

    fig.update_traces(textposition=textposition,
                      textinfo='percent+label+value',
                      texttemplate=texttemplate1,
                      marker=dict(colors=colors, line=dict(color='#ffffff', width=1))
                      )
    fig.update_layout(
        title=dict(text=title, font=dict(size=title_font_size) ), # title font size 지정
        # Add annotations in the center of the donut pies.
        annotations=[dict(text=text1, x=0.5, y=0.5, font_size=text1_font_size, showarrow=False)],
        width=figsize[0],
        height=figsize[1],
    )
    print(f"* {title}")
    return fig
    # fig.show()


def make_box_plot2(columns=[], data=[], **kwargs):
    """Box plot

    data의 길이가 각각 상이하기에, 그냥 리스트로 입력받게 함


    (Example)


    2022.4.13(Wed)"""

    colors = ['rgba(93, 164, 214, 0.5)', 'rgba(255, 144, 14, 0.5)', 'rgba(44, 160, 101, 0.5)',
              'rgba(255, 65, 54, 0.5)', 'rgba(207, 114, 255, 0.5)', 'rgba(127, 96, 0, 0.5)']

    fig = go.Figure()

    for i, each_column in enumerate(columns):
        fig.add_trace(go.Box(
            y=data[i],
            name=each_column,
            boxpoints='all',
            jitter=0.5,
            whiskerwidth=0.2,
            fillcolor=colors[i],
            marker_size=2,
            line_width=1)
        )

    fig.update_layout(
        title=kwargs.get('title', 'Title'),
        yaxis=dict(
            autorange=True,
            showgrid=True,
            zeroline=True,
            dtick=5,
            gridcolor='rgb(255, 255, 255)',
            gridwidth=1,
            zerolinecolor='rgb(255, 255, 255)',
            zerolinewidth=2,
        ),
        margin=dict(
            l=40,
            r=30,
            b=80,
            t=100,
        ),
        paper_bgcolor='rgb(243, 243, 243)',
        plot_bgcolor='rgb(243, 243, 243)',
        showlegend=False
    )
    fig.show()



def make_box_plot3(df, col1, col2,
                   title=None,
                   title_font_size=20,
                   unit=100000000,
                   unit_name='(억원)',
                   marker_color=None,
                   marker_size=3,
                   color_map=None,
                   boxpoints='outliers',
                   paper_bgcolor='#ffffff',
                   #plot_bgcolor='#ffffff',
                   plot_bgcolor='rgba(230,236,245)',
                   figsize=(800,600),
                   ):
    # 2023.5.12

    ylist = []

    x1 = df[col1].unique()
    x = sorted(x1)
    # x = range(1999,2022)
    size_x = len(list(x))

    for year in x:
        df1 = df[df[col1] == year]
        y = df1[col2] / unit  # 억단위
        ylist.append([year, y])

    fig = go.Figure()
    i = 0
    for each_year, each_y in ylist:
        i += 1
        if marker_color is None:
            c1 = find_hex_color_from_cmap(i / size_x, color_map)
        else:
            c1 = marker_color

        fig.add_trace(go.Box(
            y=each_y,
            name=f"{each_year}",
            jitter=0.3,
            pointpos=-1.8,
            # boxpoints='all', # represent all points
            # boxpoints=False, # no data points
            # boxpoints='suspectedoutliers', # only suspected outliers
            boxpoints=boxpoints,
            marker_color=c1,  # 'rgb(200,0,0)',
            marker_size=marker_size,
            line_color=c1,  # 'rgb(255,0,0)'
        ))

    fig.update_layout(
        title={'text': f"Box Plot {col2}" if title is None else title,
               'font': {'size': title_font_size}},

        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        showlegend=False,
        width=figsize[0],
        height=figsize[1],
    )

    print(f"* {title}")
    return fig

    # fig.show()
    # fig.write_image('box.pdf')

def make_histogram(df=None, name='', figsize=(800,600)):
    x0 = df.index

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x0, name=name))

    # Overlay both histograms
    fig.update_layout(barmode='overlay',
                      width=figsize[0],
                      height=figsize[1])
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    fig.show()


def make_heatmap(df,
                 title='',
                 title_font_size=20,
                 colorscale='Viridis',
                 figsize=(1000, 800),
                 margin=dict(l=0, r=50, t=100, b=50),
                 reversed=False,
                 textfont_size=8,
                 tickfont_size=8,
                 colorbar_len=0.3, # colorbar height
                 colorbar_borderwidth=0,
                 colorbar_thickness=10,
                 colorbar_orientation='v',
                 colorbar_ticklen=5,
                 colorbar_tickfont_size=5,
                 coloraxis_colorbar_x=0.5,
                 coloraxis_colorbar_y=0.1,
                 minimum_number=0.1,
                 x_min=None,
                 x_max=None,
                 ):
    #print(">>> make_heatmap()")
    textlist = []

    for eachi in df.values:
        temp = []
        for eachj in eachi:

            if eachj > minimum_number:
                eachj2 = f"{eachj:.1f}"
                temp.append(eachj2)
            else:
                temp.append('')
            #print(eachj, end=',')
        #print('')
        textlist.append(temp)

    fig=go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=[f"{each} ".upper() for each in df.index.tolist()],  # y 축
        text=textlist,
        texttemplate="%{text}",
        textfont={"size": textfont_size},
        colorscale=colorscale,  # 'reds', # Viridis'
    ))

    fig.update_layout(
        title={'text': title, 'font': {'size': title_font_size}},
        xaxis=dict(tickfont=dict(size=tickfont_size)),
        yaxis=dict(tickfont=dict(size=tickfont_size), autorange='reversed' if reversed is True else None),
        xaxis_range=[x_min, x_max],
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        margin=margin,
        coloraxis_colorbar_x=coloraxis_colorbar_x,
        coloraxis_colorbar_y=coloraxis_colorbar_y
    )
    fig.update_traces(colorbar={'outlinecolor':"#0000ff", 'bordercolor':"#ff0000",
                                'borderwidth':colorbar_borderwidth,
                                'len':colorbar_len,
                                'thickness':colorbar_thickness,
                                'orientation':colorbar_orientation,
                                 'ticklen':colorbar_ticklen,
                                'tickfont':{'size': colorbar_tickfont_size}}
                      )

    # fig.show()
    print(f"* {title}")
    return fig


def make_histogram2(df=None, colname_data=None, colname_group=None,
                    nbinsx=100,
                    title='',
                    title_sub='',
                    title_font_size=20,
                    yaxes_title=None,
                    xaxes_title=None,
                    barmode='stack',  # overlay
                    legend_size=None,
                    legend_title_size=None,
                    color_map='Set3',
                    figsize=(800,600)
                    ):
    """
    df 입력한 다음, 데이터 컬럼 이름, 그룹 컬럼 이름을 입력함


    barmode : stack 누적되어 나타남
              overlay 중첩되게 보임

    example
    ------
    critical_number = 10 # 10억원 이상
    df_raw_20_below = df_raw[df_raw['정부연구비2']<=critical_number]
    gp.make_histogram2(df=df_raw_20_below,
                   colname_data='정부연구비2',
                   colname_group='제출년도',
                   nbinsx=100,
                   legend_size=8,
                   # legend_title_size=5
                   xaxes_title="정부연구비(억원)",
                   barmode='stack',
                   title=f"정부연구비 집행액 분포 (범위 : 0 ~ {critical_number} 억원 범위)"
                  )
    """
    fig = go.Figure()
    ss1 = df[colname_group].value_counts().sort_index()

    for index in ss1.index:
        df_temp = df[df[colname_group] == index]
        x0 = df_temp[colname_data]
        # print(len(x0.index))
        fig.add_trace(go.Histogram(x=x0, nbinsx=nbinsx, name=index ))

    # Overlay both histograms
    # fig.update_layout(barmode='overlay', title="정부연구비 집행액 분포")
    fig.update_layout(barmode=barmode,
                      title={'text': title + f"<br>{title_sub}",
                             'font': {'size': title_font_size}
                             },
                      # legend=dict(font=dict(family="Courier", size=legend_size, color="black")),
                      legend=dict(font=dict(size=legend_size)),
                      legend_title=dict(font=dict(family="Courier", size=legend_title_size, color="blue")),
                      width=figsize[0],
                      height=figsize[1],
                      )
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.7)
    fig.update_yaxes(title_text='빈도수' if yaxes_title is None else yaxes_title)
    fig.update_xaxes(title_text=colname_data if xaxes_title is None else xaxes_title)

    # fig.show()
    print(f"* {title}")
    return fig


def make_graph_line(df,
                    max_option=1,
                    cum_sum=False,
                    color_map=None,
                    color_font=None,
                    xaxes_title='',
                    yaxes_title='',
                    legend='user',
                    legend_dic=None,
                    unit='',
                    marker_color="lightskyblue",
                    marker_line_width=1,
                    range_x=None,
                    range_y=None,
                    precision=1,
                    title='',
                    title_font_size=16,
                    text_cagr_font_size=5,
                    plot_bgcolor='rgba(230,236,245)',
                    node_size=3,
                    line_width=1,
                    line_colors=None,
                    textposition=None,  # top right, bottom right
                    symbols_user=None,
                    cagr=False,
                    xindex_to_string=False,
                    mode='lines+markers+text',
                    vline=False,
                    vline_text_yp=0,
                    vline_textfont_color="#ff0000",
                    hline=False,
                    hline_yp=0,
                    option=1,
                    annotation=False,
                    annotation_text='',
                    annotation_xy=(0, 0),
                    margin={},
                    width_string=100,
                    n_high=5,
                    figsize=(800,600)
                    ):
    """
    :param n_high: CAGR 표기할때 최대 개수 지정
    :return:
    """
    #print(f">>> make_graph_line(figsize={figsize}), len(df.index)={len(df.index)}")

    # 입력 데이터 확인 : 시리즈 입력하면 데이트 프레임으로 변경해 주기
    if isinstance(df, Series):
        df = DataFrame(df)

    if len(df.index) == 0:
        print(f"(warning) df.index size is 0")
        return None

    if isinstance(mode, int):
        if mode == 1:
            mode = 'lines+markers+text'
        elif mode == 2:
            mode = 'lines+markers'

    fig = go.Figure()
    # red, green, blue, yellow
    default_colors = [  # "#000000", "#ff0000", "#000000", "DarkBlue",
        # "Red", "Green", "Blue", "Gold",
        #  'Red', 'Green', 'DarkSlateBlue', 'Gold',
        'DarkRed', 'Coral', 'LightPink', 'LimeGreen',
        "#ff0000", "#00ff00", "#0000ff",
        "#ffff00", "#ff00ff", "#00ffff"]

    # symbol
    raw_symbols = SymbolValidator().values
    # print(raw_symbols)
    if symbols_user is None:
        symbols = []
        for i in range(0, len(raw_symbols), 3):
            symbols.append(raw_symbols[i + 2])
        # print(symbols)  # [ 0, '0', 'circle'  ]
        symbols1 = ['circle', 'square', 'diamond', 'cross', 'triangle-up',
                    'triangle-down']
    else:
        symbols = symbols_user

    # 사용자가 입력한 텍스트 위치 설정해주기 : textposition
    if textposition is None:
        textposition = ['top right'] * len(df.index)
    else:
        if len(textposition) != len(df.index):
            textposition = ['top right'] * len(df.index)
        else:
            pass

    # CAGR 표시
    if cagr:
        # 2022.9.14
        pearson1 = bu.make_pearson_v3(df=df, delta=0, option=option)
        pearson2 = pearson1.sort_values(by=['cagr'], ascending=[False])
        text = [f"{each['word']}: {each['cagr']:.1f}%" for index, each in pearson2.iterrows()]
        if n_high < len(text):
            pass
        else:
            n_high = len(text)
        text_cagr = ' CAGR = ' + ', '.join(text[:n_high])
        text_cagr = f'<span style="font-size: {text_cagr_font_size}px;">{text_cagr}</span>'
        text_cagr = bu.make_fixedwidth_string(txt=text_cagr, width=width_string, sep='<br>')
    else:
        text_cagr = ''

    i = 0
    x1 = df.columns

    if xindex_to_string:
        x1 = [f"{each}" for each in x1]
    names = df.index
    # print("x = ", x1)
    # print("names = ", names)

    i_color = 0
    i_line_width = 0
    i_marker_line_width = 0

    for index, y1 in df.iterrows():
        # print(index)
        y1 = y1.tolist()
        # print('y1=', y1, type(y1[0]))
        # 1) line color
        if color_map is None:
            line_color_i = default_colors[i % len(default_colors)]  # 나머지 연산자 활용 컬러 할당
            # text_color_i = line_color_i
        elif color_map == 1:  # 1 type
            line_color_i = "#000000"
        elif color_map == 2:
            if isinstance(line_colors, str):
                line_color_i = line_colors
            else:
                line_color_i = line_colors[i_color]
                i_color += 1
        else:
            # line_color_i = find_hex_color_from_cmap( y1[x1[-1]]/max2, color_map)
            line_color_i = find_hex_color_from_cmap(i / len(df.index), color_map)
            # text_color_i = line_color_i

        # 2) line width_i
        if isinstance(line_width, int | float):
            line_width_i = line_width
        elif isinstance(line_width, list):
            line_width_i = line_width[i_line_width]
            i_line_width += 1

        # 3) marker_line_width_i
        if isinstance(marker_line_width, int | float):
            marker_line_width_i = marker_line_width
        elif isinstance(marker_line_width, list):
            marker_line_width_i = marker_line_width[i_marker_line_width]
            i_marker_line_width += 1

        if color_font is not None:
            color_font1 = color_font
        else:
            color_font1 = line_color_i

        if precision == 0:
            text1 = [f"{each1:,.0f}{unit}" for each1 in y1]
        elif precision == 1:
            text1 = [f"{each1:,.1f}{unit}" for each1 in y1]
        else:
            text1 = [f"{each1:,.2f}{unit}" for each1 in y1]

        # if show_text :
        #    pass
        # else:
        #    text1 = ''
        #    print('text1=', text1)
        fig.add_trace(go.Scatter(x=x1,
                                 y=y1,
                                 mode=mode,
                                 name=names[i],
                                 text=text1,
                                 hovertemplate=f'{names[i]}',
                                 textfont={'color': color_font1, 'size': 10},
                                 textposition=textposition[i],
                                 line=dict(color=line_color_i, width=line_width_i),
                                 marker=dict(
                                     color=marker_color,  # line_color_i, #'LightSkyBlue',
                                     size=node_size,
                                     symbol=symbols[i * 4] if symbols_user is None else symbols[i],
                                     line=dict(color=line_color_i,  # 'MediumPurple',
                                               width=marker_line_width_i))))
        i += 1
    fig.update_layout(title={'text': title + '<br>' + text_cagr,
                             'font': {'size': title_font_size}
                             },
                      plot_bgcolor=plot_bgcolor,
                      paper_bgcolor='rgba(0,0,0,0)',
                      width=figsize[0],
                      height=figsize[1],
                      margin=margin,
                      )
    if annotation == True:
        fig.add_annotation(x=annotation_xy[0], y=annotation_xy[1],
                           text=annotation_text,
                           align="left",  # "right",
                           showarrow=False,
                           textangle=0,
                           xanchor='left',
                           xref="paper",
                           yref="paper")

    if vline == True:  # 최대지점 나타내 주기 (2022.9.7)
        df_y1 = df.sum(axis=0).sort_index()
        year_max = df_y1.idxmax()
        if 'sum' in df.index:
            count_max = df.loc['sum', year_max]
        else:
            count_max = vline_text_yp
        fig.add_vline(x=year_max, line_width=1, line_dash="dot", line_color="#ff0000")

        # text
        fig.add_trace(go.Scatter(
            x=[year_max],
            y=[count_max],
            mode="markers+text",
            name="Maximum Point",
            text=[f"{count_max:,.0f}"],
            textposition="top center",
            textfont=dict(
                family="sans serif",
                size=10,
                color=vline_textfont_color  # "LightSeaGreen"
            )
        ))

    if hline == True:  # 최대지점 나타내 주기 (2022.9.7)
        fig.add_hline(y=hline_yp, line_width=1, line_dash="dot", line_color="#ff0000")

    if legend_dic is not None:
        fig.update_layout(legend=dict(
            yanchor=legend_dic.get('yanchor', "bottom"),
            y=legend_dic.get('y', -0.2),
            xanchor=legend_dic.get('xanchor', "center"),
            x=legend_dic.get('x', 0.5),
            orientation=legend_dic.get('orientation', "h"),
            font=dict(  # family = "Courier",
                size=legend_dic.get('font_size', 10),
                color='black',  # legend_dic('font_color', "black"
            )
        )
        )

        fig.update_layout(legend_traceorder="reversed")

    if range_x is not None:
        fig.update_xaxes(title_text=xaxes_title, range=range_x)
    else:
        fig.update_xaxes(title_text=xaxes_title)
    if range_y is not None:
        fig.update_yaxes(title_text=yaxes_title, range=range_y)
    else:
        fig.update_yaxes(title_text=yaxes_title)
    # fig.show()
    #print("<< make_grpah_line()")
    print(f"* {title}")
    return fig


def make_graph_bar(df,
                   title='',
                   xaxes_title_text='',
                   yaxes_title_text='',
                   tickfont_size=10,
                   orientation='v',
                   user_color=None,
                   color_map=None,
                   width=None,
                   precision=0,
                   default_width=1,
                   unit=''):
    """ df 입력받아서, 인덱스를 x 축으로 하는 세로 막대 그래프 그리기,

    바 색을 한가지 색으로 할것인가 ? 여러가지 색으로 할 것인가 ?
    user_color 주어지면 그대로 하기
    또는 user_color 가 None일때 color_map 주어지면, color_map에 따라 지정하기

    2022.5.5 precision 추가


    """
    # 시리즈 입력하면 데이트 프레임으로 변경해 주기
    if isinstance(df, Series):
        df = DataFrame(df)

    if width is None:
        width = [default_width for each in range(len(df.index))]

    if user_color is not None:
        if isinstance(user_color, str):
            bar_colors = user_color * len(df.index)
        elif isinstance(user_color, list):
            bar_colors = user_color
    else:
        if color_map is None:
            # bar_colors = default_colors[i%len(default_colors)] # 나머지 연산자 활용 컬러 할당
            # text_color_i = line_color_i
            pass
        else:
            # line_color_i = find_hex_color_from_cmap( y1[x1[-1]]/max2, color_map)
            # text_color_i = line_color_i
            bar_colors = [find_hex_color_from_cmap(i / len(df.index), color_map) for i in range(len(df.index))]
    data = []
    if orientation == 'v':
        print("orientation = v")
        if user_color is not None:
            j = 0
            for each_column in df.columns:
                data.append(go.Bar(x=df.index,
                                   y=df[each_column],
                                   name=each_column,
                                   text=[f"{each:,.{precision}f}{unit}" for each in df[each_column]],
                                   orientation=orientation,
                                   marker_color=user_color[j],
                                   )
                            )
                j += 1
        else:
            j = 0
            for each_column in df.columns:
                data.append(go.Bar(x=df.index,
                                   y=df[each_column],
                                   width=width,
                                   name=each_column,
                                   text=[f"{each:,.{precision}f}{unit}" for each in df[each_column]],
                                   orientation=orientation,
                                   )
                            )
                j += 1
        fig = go.Figure(data=data)
        fig.update_layout(title=title,
                          height=500,
                          # yaxis=dict(autorange="reversed")
                          xaxis=dict(tickfont=dict(size=tickfont_size)),
                          )
        fig.update_xaxes(title_text=xaxes_title_text)
        fig.update_yaxes(title_text=yaxes_title_text, side='left', )
    else:
        print("orientation is not vertical")
        for each_column in df.columns:
            data.append(go.Bar(x=df.index,
                               y=df[each_column],
                               name=each_column,
                               text=[f"{each:.1f}" for each in df[each_column]],
                               orientation=orientation,
                               )
                        )
        fig = go.Figure(data=data)
        fig.update_layout(title=title,
                          height=500,
                          # yaxis=dict(autorange="reversed")
                          xaxis=dict(tickfont=dict(size=tickfont_size)),
                          )
        fig.update_xaxes(title_text=xaxes_title_text)
        fig.update_yaxes(title_text=yaxes_title_text, side='left', )
    fig.show()


def make_graph_bar2(df=None,
                    values=None,
                    index=None,
                    title='',
                    title_sub='',
                    title_font_size=20,
                    textposition='auto',
                    textfont_size=None,
                    textfont_color=None,
                    xaxes_title='',
                    yaxes_title='',
                    tickfont_size=10,
                    orientation='v',
                    user_color=None,
                    color_map=None,
                    width=None,
                    precision=0,
                    default_width=0.8,
                    barmode='stack',  # stack or group
                    unit='',
                    marker_line_width=.5,
                    opacity=1.0,
                    to_percent=False,
                    plot_bgcolor=None,
                    paper_bgcolor=None,
                    height=None,
                    reversed=False,
                    figsize=(800, 600), # width, height
                    annotation=True,
                    annotation_text='',
                    annotation_xy=(0, 0),
                    margin={},
                    legend_title_text='',
                    range_x=None,
                    range_y=None,
                    ):
    """
    df 입력 받아서, 인덱스를 x 축으로 하는 세로 막대 그래프 그리기,
    바 색을 한 가지 색으로 할 것인가 ? 여러 가지 색으로 할 것인가 ?
    user_color 주어지면 그대로 하기
    또는 user_color 가 None 일 때 color_map 주어지면, color_map에 따라 지정하기

    2022.5.5 precision 추가
    2022.5.5 함수 이름 변경

    Example
    ------
    df_test1 = DataFrame([1,2,3], index=['a','b','c'])
    fig = gp.make_graph_bar2(df=df_test1,
                             width=0.5,
                             unit='개',
                             title="사업화 데이터 : 1) 원본 2) 분석대상 3) 제외대상",
                             user_color=[ ['#1f77b4','#ff7f02','#2ca02c']],
                             yaxes_title='데이터 개수')
    """
    #print(f">>> make_graph_bar2(figsize={figsize})")
    # 입력 데이터 확인 : 시리즈 입력하면 데이트 프레임으로 변경해 주기
    if df is None:
        if (values is not None) and (index is not None):

            df = DataFrame(values, index=index, columns=['data1'])
            # print('# df is None, values and index is given :', df.shape)
            # print('# length = ', len(df.index))
            # print(df)
        else:
            # print("# check your input data : df !")
            return None

    if isinstance(df, Series):
        df = DataFrame(df)

    # 비율로 나타낼 것인지 확인
    if to_percent:
        dfp = 100 * df.div(df.sum(axis=1), axis=0)
        df = dfp

    # 폭 설정
    if width is None:
        # width = [default_width for each in range(len(df.index))]
        width = [default_width for each in range(len(df.columns))]
    else:
        # scalar 이면, list로 변경
        # print(" scaler 이면 list로 변경")
        if isinstance(width, (int, float)):
            width = [width] * len(df.columns)
            # print('width=', width)

    # 바 색 설정 : 사용자가 직접 설정해주는 방법 : 문자열 일 경우, 리스트 일 경우.
    if user_color is not None:
        if isinstance(user_color, str):
            marker_colors = [user_color] * len(df.columns)
            #print(f"case A1 : marker_colors={marker_colors}")
        elif isinstance(user_color, list):
            marker_colors = user_color
            #print(f"case A2 : marker_colors={marker_colors}")
            # print('user_color=', marker_colors)
        else:
            marker_colors = ['#eeeeee'] * len(df.columns)
            #print(f"case A3 : marker_colors={marker_colors}")
    else:
        # 사용자 색 지정하지 않았다면,
        if color_map is None:
            # bar_colors = default_colors[i%len(default_colors)] # 나머지 연산자 활용 컬러 할당
            # text_color_i = line_color_i
            marker_colors = ['#1f77b4',  # // muted blue
                             '#ff7f0e',  # // safety orange
                             '#2ca02c',  # // cooked asparagus green
                             '#d62728',  # // brick red
                             '#9467bd',  # // muted purple
                             '#8c564b',  # // chestnut brown
                             '#e377c2',  # // raspberry yogurt pink
                             '#7f7f7f',  # // middle gray
                             '#bcbd22',  # // curry yellow-green
                             '#17becf'  # // blue-teal
                             ] + ['#17bedf'] * len(df.columns)  # 에러 방지용 : IndexError: list index out of range
            #print(f"case B1 : marker_colors={marker_colors}")
        else:
            # print("color_map=", color_map)
            # line_color_i = find_hex_color_from_cmap( y1[x1[-1]]/max2, color_map)
            #n_size = len(df.index)
            n_size = len(df.columns)
            marker_colors = [find_hex_color_from_cmap( (i+1)/n_size , color_map) for i in range(n_size)]
            #print(f"case B2 : marker_colors={marker_colors}")
            # text_color_i = line_color_i

    data = []
    x1 = [f"{each}" for each in df.index]  # x축 , 정수 일 때 여러개 틱 레이블 나타나는 문제 해결
    if orientation == 'v':
        # print("orientation = v")
        j = 0
        for each_column in df.columns:
            # print(f"orientation = v, marker_colors[{j}]=", marker_colors[j])
            # print('*** each_column =', each_column, df[each_column])
            data.append(go.Bar(x=x1,  # df.index,
                               y=df[each_column],
                               name=each_column,
                               #text=[f"{each_column}<br>{each:,.{precision}f}{unit}" for each in df[each_column]],
                               text=[f"{each:,.{precision}f}{unit}" for each in df[each_column]],
                               orientation=orientation,
                               marker_color=marker_colors[j],
                               textposition=textposition,
                               textfont={'color': textfont_color, 'size': textfont_size},
                               width=width[j],
                               )
                        )
            j += 1
        xaxes_title1 = xaxes_title
        yaxes_title1 = yaxes_title
    else:
        #print("orientation is not vertical - horizontal, 수평")
        # print(df)
        j = 0
        for each_column in df.columns:
            data.append(go.Bar(y=x1,
                               x=df[each_column],
                               name=each_column,
                               text=[f"{each:,.{precision}f}{unit}" for each in df[each_column]],
                               orientation=orientation,
                               width=width[j],
                               marker_color=marker_colors[j],
                               textposition=textposition
                               )
                        )
            j += 1
        xaxes_title1 = yaxes_title
        yaxes_title1 = xaxes_title

    fig = go.Figure(data=data)
    fig.update_traces(
        marker_line_color='#ffffff',
        marker_line_width=marker_line_width,
        opacity=opacity)
    fig.update_layout(# title=dict(text=title, font=dict(size=title_font_size)), # title font size
                      title={'text': title + f"<br>{title_sub}",
                             'font': {'size': title_font_size}
                             },
                      # height= height if height is not None else 500,
                      # yaxis=dict(autorange="reversed")
                      xaxis=dict(tickfont=dict(size=tickfont_size)),
                      yaxis=dict(tickfont=dict(size=tickfont_size), autorange='reversed' if reversed is True else None),
                      barmode=barmode,
                      paper_bgcolor=paper_bgcolor,
                      plot_bgcolor=plot_bgcolor,
                      width=figsize[0],
                      height=figsize[1],
                      margin=margin,
                      legend_title_text=legend_title_text,
                      )
    if annotation == True:
        fig.add_annotation(x=annotation_xy[0], y=annotation_xy[1],
                           text=annotation_text,
                           align="left",  # "right",
                           showarrow=False,
                           textangle=0,
                           xanchor='left',
                           xref="paper",
                           yref="paper")
    fig.update_xaxes(title_text=xaxes_title1)
    fig.update_yaxes(title_text=yaxes_title1, side='left', )

    if range_x is not None:
        fig.update_xaxes(range=range_x)
    if range_y is not None:
        fig.update_yaxes(range=range_y)

    # fig.show()
    print(f"* {title}")
    return fig


def make_graph_horizontal_bar(df=None,
                              title='',
                              width=1800,
                              height=1800,
                              xaxes_title_text='',
                              yaxes_title_text='',
                              font_size=14,
                              font_size_y=14,
                              paper_bgcolor='rgb(248, 248, 255)',
                              plot_bgcolor='rgb(248, 248, 255)',
                              barmode='stack',
                              margin=dict(l=0, r=50, t=100, b=50),
                              inside_unit='%',
                              precision=1,
                              ):
    """
    2022.4.26
    reference : https://plotly.com/python/horizontal-bar-charts/

    2022.12.2 수정

    """
    top_labels = df.columns
    colors = [find_hex_color_from_cmap(i / len(top_labels)) for i in range(len(top_labels))]
    x_data = df.values
    # y_data = [int(each) for each in df.index]
    y_data = [each for each in df.index]
    fig = go.Figure()

    for i in range(0, len(x_data[0])):
        for xd, yd in zip(x_data, y_data):
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=2)
                )
            ))
    fig.update_layout(
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
            domain=[0.15, 1],
        ),
        yaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),

        barmode=barmode,
        paper_bgcolor=paper_bgcolor,
        plot_bgcolor=plot_bgcolor,
        margin=margin,
        showlegend=False,
    )
    annotations = []
    for yd, xd in zip(y_data, x_data):
        # labeling the y-axis : y축 라벨링
        annotations.append(dict(xref='paper', yref='y',
                                x=0.1, y=yd,
                                xanchor='right',
                                text=f"{yd}",  # str(yd),
                                font=dict(family='Arial',
                                          size=font_size_y,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False, align='right'))
        # labeling the first percentage of each bar (x_axis)
        annotations.append(dict(xref='x', yref='y',
                                x=xd[0] / 2, y=yd,
                                text=f"{xd[0]:,.{precision}f} {inside_unit}",
                                font=dict(family='Arial',
                                          size=font_size,
                                          color='rgb(248, 248, 255)'),
                                showarrow=False))
        # labeling the first Likert scale (on the top)
        if yd == y_data[-1]:
            annotations.append(dict(xref='x', yref='paper',
                                    x=xd[0] / 2, y=1.05,
                                    text=top_labels[0],
                                    font=dict(family='Arial',
                                              size=font_size,
                                              color='rgb(67, 67, 67)'),
                                    showarrow=False))
        space = xd[0]
        for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i] / 2), y=yd,
                                    text=f"{xd[i]:,.{precision}f} {inside_unit}",  # str(xd[i]) + '%',
                                    font=dict(family='Arial',
                                              size=font_size,
                                              color='rgb(248, 248, 255)'),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i] / 2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial',
                                                  size=font_size,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]
    fig.update_layout(title=title, annotations=annotations, width=width, height=height,
                      font=dict(
                          family="Courier New, monospace",
                          size=18,
                          color="RebeccaPurple")
                      )
    fig.update_xaxes(title_text=xaxes_title_text)
    fig.update_yaxes(title_text=yaxes_title_text, side='left')
    # fig.show()
    return fig


# 저장
@deco1
def save_fig_to_pdf(fig_total: list,
                    fig_name_prefix='',
                    year_begin='',
                    year_end='',
                    rank_n='',
                    n_newness='') -> str:
    """plotly figures to pdf file

    fig_total : fig 리스트

    """

    temp_files = [f'{fig_name_prefix}_{i + 1}.pdf' for i in range(len(fig_total))]
    for each_fig, each_file in zip(fig_total, temp_files):
        each_fig.write_image(each_file)

    # fname2 = f'{fig_name_prefix}_{year_begin}_{year_end}_top_{rank_n}_newness_{n_newness}.pdf'
    fname2 = f'{fig_name_prefix}.pdf'
    bu.merge_pdf(temp_files, fname2, remove=True)
    return fname2


def make_history_graph(df=None, col_name='name', col_birth='birth', col_death='death', title='',
                       color="#eeeeee", color_text="#ffffff", figsize=(400, 400)):
    """
    history graph
    """
    print(figsize)
    fig = go.Figure()

    yp = 0
    for index, each in df.iterrows():
        # print(each)

        yp += 1
        fig.add_trace(go.Scatter(y=[yp, yp],
                                 # y=[index,index],
                                 # x=[f"{each[col_birth]} 00:00:00",f"{each[col_death]} 00:00:00"],
                                 x=[each[col_birth], each[col_death]],
                                 mode="lines+text",
                                 line=dict(color=color, width=20),
                                 text=[f"{each[col_name]}", ""],
                                 textfont={"color": [color_text, color_text],
                                           "size": [10, 10]},
                                 textposition="middle right",
                                 name=each[col_name]
                                 )

                      )
    # fig.update_layout(title=title)
    fig.update_layout(
        title=title,
        width=figsize[0],
        height=figsize[1],
        showlegend=False,
        # margin=margin,
    )
    return fig


if __name__ == '__main__':
    print(">>> ")
