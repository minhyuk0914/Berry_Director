from flask import Flask, render_template, request, redirect, url_for, Blueprint, app
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import shutil
import numpy as np
import torch
import qrcode

# 서버 열기
# set FLASK_APP=app.py
# flask run --host=0.0.0.0 --port=5000

model = YOLO('best.pt')




def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    
app = Flask(__name__)



@app.route("/")
def index():
    # runs 디렉토리 삭제
    try:
        shutil.rmtree('static/images/runs')
    except:
        print('index_파일이 없습니다.')

    # strawberry 디렉토리 삭제
    try:
        shutil.rmtree('static/images/strawberry/')
    except:
        print('index_디렉토리가 없습니다.')


    # qrcode 디렉토리 삭제
    try:
        shutil.rmtree('static/images/qrcode/')
    except:
        print('index_디렉토리가 없습니다.')
    
    return render_template('index.html')

@app.route("/image_upload")
def image_upload():

    # runs 디렉토리 삭제
    try:
        shutil.rmtree('static/images/runs')
    except:
        print('image_upload_파일이 없습니다.')

    # strawberry 디렉토리 삭제
    try:
        shutil.rmtree('static/images/strawberry/')
    except:
        print('image_upload_디렉토리가 없습니다.')

    # qrcode 디렉토리 삭제
    try:
        shutil.rmtree('static/images/qrcode/')
    except:
        print('image_upload_디렉토리가 없습니다.')
    
    return render_template('image_upload.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/uploader", methods=['GET','POST'])
def uploader():
    if request.method == 'POST':
        # strawberry 디렉토리 생성
        try:
            os.makedirs("static/images/strawberry/")
        except:
            print('디렉토리가 있습니다.')

        # qrcode 디렉토리 생성
        try:
            os.makedirs("static/images/qrcode/")
        except:
            print('디렉토리가 있습니다.')

        # 사용자 이미지 저장
        user_img_name= []

        user_img = request.files.getlist("user_img[]")

        for file in user_img:
            file.save('static/images/strawberry/'+file.filename)
            user_img_name.append(file.filename)

        # print("유저 이미지: ",user_img_name)
    

        # 이미지 모델 돌리기
        total = []
        ripe = []
        no_ripe = []
        

        for file in user_img_name:
            model_img = model.predict('static/images/strawberry/'+file, save = True, hide_conf = True)
            model_img_path = 'runs/detect/predict/'+str(model_img)
            # 이미지 개수 구하기
            for r in model_img:
                total.append(r.__len__())
                ripe.append((torch.count_nonzero(r.boxes.cls)).item())
                no_ripe.append(r.__len__() - (torch.count_nonzero(r.boxes.cls)).item())


        # 리스트 채우기
        user_img_name_2 = ['0','1','2','3','4']
        user_img_name_3 = user_img_name + user_img_name_2
        
        total_2 = total + [0,1,2,3,4]
        ripe_2 = ripe + [0,1,2,3,4]
        no_ripe_2 = no_ripe + [0,1,2,3,4]


        # print("유저 이미지 3: ",user_img_name_3)


        shutil.move('runs','static/images')


        img_len = len(total)
        # print("전체 개수: ", total)
        # print("익은 개수: ", ripe)
        # print("안 익은 개수: ", no_ripe)
        # print("리스트 길이: ", img_len)
        # print("리스트 타입: ", type(img_len))
        

        # QR 코드 만들기
        host_url = "http://3.39.208.236:1004/"
        for name in user_img_name:
            qr_img = qrcode.make(host_url + "/static/images/runs/detect/predict/" + name)
            qr_img.save('static/images/qrcode/'+name)



        
    return render_template('information.html', user_img_name = user_img_name_3, total = total_2, ripe = ripe_2, no_ripe = no_ripe_2, img_len=img_len)
                                                
                                                 
        


@app.route("/information")
def information():

    return render_template('information.html')


if __name__ == '__main__':
    app.debug = True
    app.run(host="0.0.0.0", port="1004")