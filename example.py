# test for bkNTIS package
import time
from bkntis import bk_ntis as bn

t1 = time.time()

filename1 = '/Users/bk/Documents/ntis_python/ntis_1999_2021.pkl'

ntis1 = bn.NTIS(filename=filename1)

on1 = [
       #'null',
       #'fund_trend1',
       #'fund_histogram',
       #'fund_boxplot',
       #'fund_trend2',
       'col_groups'
      ]

col_groups1 = [
    #'사업_부처명',
               '사업_부처명m',
             #  '연구수행주체',
             #  '지역', '지역2', '지역3',
             #  '6T관련기술-대', '과학기술표준분류1-대', '중점과학기술분류-대',
             #  '연구개발단계',
             #  '연구책임자성별',
             #  '보안과제3',
             #  '연구비_등급1', '연구비_등급2', '연구비_등급4'
               ]

fig_list = ntis1.make_pdf(filename_prefix='ministry2', # 한글 작성시 조심
                          #col_groups_fund=['연구비_등급1', '연구비_등급2', '연구비_등급4'],
                          col_groups=col_groups1,
                          color_map='Set3',

                          onoff_condition=on1)

t2 = time.time() - t1

print(f"{t2:.1f} seconds !")
