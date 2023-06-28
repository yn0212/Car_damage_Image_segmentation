# In[18]:
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import  os
import subprocess
import sys
sys.path.append('D:/jyn/ncslab/seg_tensorflow/seg_tfds_generator/seg_tfds_module.py')
from seg_tfds_module import *
#경로 설정
work_path='D:/jyn/ncslab/data/car0623' #main 작업 폴더 설정
json_path = 'D:/jyn/ncslab/data/damage_label' # json파일
images_path = 'D:/jyn/ncslab/data/damage_image' # 이미지


#데이터셋 개수 선택 후 이미지, json파일 추출
select_data(images_path,json_path,work_path,10)

#json파일 추출 후 형식 변경 ,JSON 파일 형식 안맞는 파일 삭제
make_json_file(work_path)


#아나콘다 터미널에서 labelme명령어 실행 후 dataset생성하기
make_dataset(work_path)


#trimap이미지 생성
make_trimap(work_path , 10)

#trimap 객체 경계 배경 구분하는 픽셀값 지정
convert_trimap_pixel(work_path)

#json파일 파싱해서 형식에 맞는 xml파일 구성
make_xml(work_path)

#전체 데이터셋의 이미지 파일 이름과 클래스 레이블 정보담긴 txt파일 생성
make_txt_file(work_path)


#훈련데이터 검증데이터 설정 7:3비율로 나누기
split_text_file(work_path ,0.7)

# 폴더 생성
create_folder(work_path)

#데이터셋 폴더 압축
create_tar_gz(work_path)

# %%
