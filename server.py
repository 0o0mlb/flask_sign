from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from src import model
from src import util
from src.body import Body
from src.hand import Hand

from sklearn.preprocessing import Normalizer
import tensorflow as tf

modelpt = 'model'
body_estimation = Body('model/body_pose_model.pth')
hand_estimation = Hand('model/hand_pose_model.pth')

app = Flask(__name__)


class OpenPose():
   def __init__(self, modelpath):
      self.body_estimation = Body(modelpath + '/body_pose_model.pth')
      self.hand_estimation = Hand(modelpath + '/hand_pose_model.pth')

   def handpt(self, oriImg):
      candidate, subset = self.body_estimation(oriImg)
      canvas = copy.deepcopy(oriImg)
      canvas = util.draw_bodypose(canvas, candidate, subset)

      hands_list = util.handDetect(candidate, subset, oriImg)
      all_hand_peaks = []
      cnt = 0

      for x, y, w, is_left in hands_list:
         peaks = self.hand_estimation(oriImg[y:y + w, x:x + w, :])
         peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
         peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)

         all_hand_peaks.append(peaks)

      return all_hand_peaks

@app.route('/')
def render_file():
   return render_template('main.html')

@app.route('/model', methods = ['GET', 'POST'])
def model():
   if request.method == 'POST':
      f = request.files['file']
      #저장할 경로 + 파일명
      f.save(secure_filename(f.filename))

      pose = OpenPose(modelpt)
      arr = []
      temp_peaks = []

      vidcap = cv2.VideoCapture(f.filename)
      count = 0

      while (vidcap.isOpened()):
         ret, image = vidcap.read()
         if not ret:
            break

         # 영상 길이에 따라 이미지 추출 간격 조절
         total_frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
         frame = int(total_frame_count / 20)

         # frame프레임당 하나씩 이미지 추출
         if (int(vidcap.get(1)) % frame == 0 and count < 20):
            position = pose.handpt(image)
            out_arr = np.ravel(position, order='C')  # 1차원으로 만들기(0:84)

            if out_arr.size == 0:
               print("No detect. 모든 keypoint zero 처리")
               temp_peaks = np.zeros(84)
               out_arr = temp_peaks.reshape(1, 84)

            if out_arr.size == 42:
               print("No detect. 이전 keypoint 불러오기")
               out_arr = temp_peaks.reshape(1, 84)

            if out_arr.size == 84:
               temp_peaks = out_arr
               print("Detect 성공!")
               out_arr = out_arr.reshape(1, 84)

            # 데이터 한 줄로 만들고 dataframe 생성
            arr = np.append(arr, out_arr)
            count += 1

      vidcap.release()

      X = arr.reshape(1, 1680)
      transformer = Normalizer().fit(X)
      X = transformer.transform(X)

      model = tf.keras.models.load_model('model_100.h5')
      y_predict = model.predict(X)
      label = y_predict[0].argmax()

      CLASSES = ["춥다", "덥다", "먹구름", "바람", "비", "온도", "장마", "햇빛", "싫다", "좋다", "양산", "체온", "하늘", "강풍", "따뜻하다", "홍수", "예정",
                 "사계절", "눈", "뙤약볕",
                 "감기", "에어컨", "체감", "경고", "찻길", "출근", "강", "파도", "어젯밤", "방금", "어떻게", "자정", "아까", "최대", "최소", "평일",
                 "깜깜하다",
                 "틀림없다", "매일매일", "혹시",
                 "들어맞다", "사라지다", "오래도록", "찰나", "이미", "순식간", "모르다", "불가능", "충분", "불안", "당황", "원하다", "무섭다", "빠르다", "즐겁다",
                 "마지막", "오다", "깨끗하다", "행복", "떨다",
                 "조심", "변덕", "망설이다", "희망", "힘들다", "화나다", "놀라다", "사람", "우울", "빨리", "못하다", "시원하다", "기억", "무지개", "불신",
                 "대략",
                 "피곤", "시작", "충격", "조용하다",
                 "상상", "예견", "난감하다", "알려주다", "잊어버리다", "깜빡하다", "결코", "제법", "뜻밖", "격노", "짜증내다", "강물", "달", "뜨겁다", "별",
                 "호수",
                 "일몰", "일출", "적중하다", "정말"]

      #return '해당 영상은 ' + CLASSES[label] + ' 입니다.'
      return render_template('result.html', label=CLASSES[label])


if __name__ == '__main__':
    app.run(debug=True)
