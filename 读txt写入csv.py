# -*- coding: UTF-8 -*-
import os,os.path
import csv

with open('corpus.csv', 'w',encoding='utf-8',newline='') as f:
    csv_writer = csv.writer(f)
    files = os.listdir('./')
    datas = []
    for filename in files:
        if os.path.isdir(filename) and filename!='models':
            category = filename
            for file in os.listdir('./'+filename):
                path = './'+filename+'/'+file
                r = open(path, 'r',encoding='utf-8')
                line_a = r.readlines()
                text = ''
                for line in line_a[2:]:
                    if line==' 'or line=='\n':
                        continue
                    line = line.replace("\t", "")
                    # line = line.replace(",", "，")
                    text += line.strip()
                r.close()
                datas.append([text,category])
    csv_writer.writerows(datas)
