from IPython import get_ipython
from trimap_module import trimap
import cv2, os, sys
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
import os
import base64
import xml.etree.ElementTree as ET
import shutil
#입력한 개수의 이미지파일 ,json파일 선택
def select_data(image_path,json_path,work_path,num_files):
    # 복사할 폴더 경로 , 폴더 생성
    destination_img_folder = os.path.join(work_path,'images')
    os.makedirs(destination_img_folder, exist_ok=True)
    destination_json_folder =os.path.join(work_path,'json')
    os.makedirs(destination_json_folder,exist_ok=True)
    # 원본 폴더 내 파일 목록 가져오기
    img_files = os.listdir(image_path)
    json_files =os.listdir(json_path)
    # 파일 이름을 기준으로 정렬
    img_files_sorted = sorted(img_files)
    json_files_sorted =sorted(json_files)

    # 일정 개수의 이미지파일을 복사하고 목적 폴더로 붙여넣기
    for i, file_name in enumerate(img_files_sorted):
        if i == num_files:  # 지정한 개수의 파일을 추출했으면 반복문 종료
            break
        
        source_file = os.path.join(image_path, file_name)  # 원본 파일 경로
        destination_file = os.path.join(destination_img_folder, file_name)  # 목적 파일 경로
        
        shutil.copy(source_file, destination_file)  # 파일 복사
    # 일정 개수의 json파일을 복사하고 목적 폴더로 붙여넣기
    for i, file_name in enumerate(json_files_sorted):
        if i == num_files:  # 지정한 개수의 파일을 추출했으면 반복문 종료
            break
        
        source_file = os.path.join(json_path, file_name)  # 원본 파일 경로
        destination_file = os.path.join(destination_json_folder, file_name)  # 목적 파일 경로
        
        shutil.copy(source_file, destination_file)  # 파일 복사
    

#json파일 추출 후 형식 변경
def make_json_file(work_path ):
    json_folder_path = os.path.join(work_path,'json')
    img_folder_path =os.path.join(work_path,'images')
    print(img_folder_path)
    for x in os.listdir(json_folder_path):
        #이미지 파일 경로
        image_file_name= x.replace('.json','.jpg')
        
        image_path = os.path.join(img_folder_path,image_file_name)
        # 기존 JSON 파일 경로
        json_file_path = os.path.join(json_folder_path,x)
        
        # 이미지 데이터를 Base64로 인코딩
        with open(image_path, "rb") as f:
            image_data = f.read()
            encoded_image_data = base64.b64encode(image_data).decode("utf-8")
        
        # 기존 JSON 파일 열기
        with open(json_file_path, "r") as f:
            data = json.load(f)

        # 필요한 값을 가져와서 새로운 JSON 데이터 생성
        # "shapes" 필드에서 "points"와 "label" 값 가져오기
        ex =0
        shapes_list = []
        for shape in data['annotations']:
            points = shape['segmentation']
            label = shape['damage']
            bbox = shape['bbox']
            
            try:
                points =np.array(points)
                points = np.reshape(points,(points.shape[2],-1))
                points = points.tolist()
            except:
            # JSON 파일 파싱 오류 또는 파일이 존재하지 않으면 이미지와 json삭제(이미지파일도 함께)
                for y in os.listdir(img_folder_path) :
                    file_name= x
                    #이미지파일 삭제
                    if y[:-4]== file_name[:-5]:
                        img_path =os.path.join(img_folder_path,y)
                        os.remove(img_path)
                os.remove(os.path.join(json_folder_path,x))
                print(f"File '{os.path.join(json_folder_path,x)}' deleted due to JSON parsing error or file not found.")
                ex=1
                break


            shape_data = {
                'points': points,
                'label': label,
                'bbox' : bbox,
                "shape_type": "polygon"
            }
            shapes_list.append(shape_data)

        img_Path=data['images']['file_name']
        img_h=data['images']['height']
        img_w=data['images']['width']
        # "shapes" 필드에 "points"와 "label" 값들 넣기
        new_data = {
            'shapes': shapes_list,
            'imagePath': img_Path,
            'imageData':encoded_image_data,
            'imageHeight' : img_h,
            'imageWidth' : img_w
        }

        json_new_path =os.path.join(work_path,'json_new')
        os.makedirs(json_new_path, exist_ok=True)
        if ex==1 : 
            continue
        else:

            # 수정된 JSON 파일 저장            
            output_json_file_path = os.path.join(json_new_path ,x) 
            with open(output_json_file_path, "w") as f:
                json.dump(new_data, f)


def make_dataset(work_path):
    #아나콘다 환경에서 labelme_json_to_dataset 명령어 입력
    cwd=os.getcwd()
    shutil.copy(os.path.join(cwd,'json_to_dataset.py'), os.path.join(work_path,'json_to_dataset.py'))
    conda_env_name = "tensorflow"
    work_path = "D:/jyn/ncslab/data/car0623"

    conda_command = f"conda activate {conda_env_name}"
    command = 'python json_to_dataset.py json_new -o labelme'

    subprocess.run(f"conda run -n {conda_env_name} {command}", shell=True,cwd=work_path)


#json이미지 픽셀 변환 후 trimap 이미지로 변환        
def make_trimap(work_path ,thickness_size):
    json_image_path = os.path.join(work_path,'labelme')
    trimap_path = os.path.join(work_path,'trimap')
    #폴더 생성
    os.makedirs(trimap_path, exist_ok=True)
    for x in os.listdir(json_image_path):
        
        image= cv2.imread(json_image_path+'/'+x, cv2.IMREAD_GRAYSCALE) #불러오기
        h,w =image.shape
        for i in range(h): #labelme 이미지 픽셀값 변경 
            for j in range(w):
                if(image[i,j]== 0):
                    image[i,j]=0
                else :
                    image[i,j]=255
        size = thickness_size;         #ex)10 Unknown Region Thickness
        number = x[-5]; # Obtain The Image Number (in case more than one image)
        name=os.path.join(trimap_path,x)
        height, width = image.shape[:2]
        trimap(image, name, size, number) #trimap변환

#trimap 객체 경계 배경 구분하는 픽셀값 지정
def convert_trimap_pixel(work_path  ):
    trimap_path=os.path.join(work_path,'trimap')
    path = os.path.join(work_path,'trimaps') ## Change the directory
    os.makedirs(path, exist_ok=True)
    for x in os.listdir(trimap_path):
        image_path = os.path.join(trimap_path, x)
        image= cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) #불러오기
        h,w =image.shape
        for i in range(h):              #픽셀값 변경
            for j in range(w):
                if(image[i,j]== 0):     #배경이면 2
                    image[i,j]=2
                elif(image[i,j]==255) : #객체이면 1
                    image[i,j]=1
                else:                   #경계이면 3
                    image[i,j]=3
        
        cv2.imwrite(os.path.join(path, x) , image)

#json파일 파싱해서 형식에 맞는 xml파일 구성
def make_xml(work_path ):
    json_path = os.path.join(work_path,'json_new')
    for json_file in os.listdir(json_path):
        path = os.path.join(json_path,json_file)
        with open(path, "r") as file:
            json_data = json.load(file)

        # JSON 데이터 파싱
        imgpath = json_data["imagePath"]

        #flags = json_data["flags"]
        shapes = json_data["shapes"]
        imgheight=json_data["imageHeight"]
        imgwidth=json_data["imageWidth"]

        # 루트 요소 생성
        root = ET.Element("annotation")

        # 하위 요소 추가
        folder = ET.SubElement(root, "folder")
        folder.text = "CAR"

        filename = ET.SubElement(root, "filename")
        filename.text = imgpath

        database =ET.SubElement(root, "database")
        database.text = 'CAR Dataset'

        annotation =ET.SubElement(root, "annotation")
        annotation.text = "CAR"

        image =ET.SubElement(root, "image")
        image.text = "flickr"

        width =ET.SubElement(root, "width")
        width.text =str(imgwidth)

        height =ET.SubElement(root, "height")
        height.text =str(imgheight)

        depth =ET.SubElement(root, "depth")
        depth.text ='3'

        segmented =ET.SubElement(root, "segmented")
        segmented.text = '0'

        # 필요한 데이터 추출
        for shape in shapes:

            label = shape["label"]
            points = shape["points"]
            bbox_val = shape["bbox"]

            # 추가적인 처리 작업 수행
            object1 = ET.SubElement(root, "object")

            name = ET.SubElement(object1, "name")
            name.text =label
            pose =ET.SubElement(object1, "pose")
            pose.text = 'Frontal'

            truncated=ET.SubElement(object1, "truncated")
            truncated.text='0'
            occluded =ET.SubElement(object1, "occluded")
            occluded.text='0'

            bndbox=ET.SubElement(object1, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text=str(bbox_val[0])
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text=str(bbox_val[1])
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text=str(bbox_val[2])
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text=str(bbox_val[3])

            d = ET.SubElement(bndbox, "difficult")
            d.text = '0'

            # XML 파일 저장
            tree = ET.ElementTree(root)
            output_path=os.path.join(work_path,'xmls')
            os.makedirs(output_path, exist_ok=True)
            xml_file_path = output_path+'/' +imgpath[:-4]+'.xml'
            tree.write(xml_file_path)

#전체 데이터셋의 이미지 파일 이름과 클래스 레이블 정보 txt
def make_txt_file(work_path ):
    json_file_path=os.path.join(work_path,'json_new')
    text_file_path=os.path.join(work_path,'list.txt')
    from collections import Counter
    for json_file in os.listdir(json_file_path):
        path = os.path.join(json_file_path, json_file)
        with open(path, "r") as file:
            json_data = json.load(file)

        # JSON 데이터 파싱
        imgpath = json_data["imagePath"]
        imgpath= imgpath[:-4]
        #flags = json_data["flags"]
        shapes = json_data["shapes"]

        pre_label=[]
        # 필요한 데이터 추출
        for shape in shapes:
            label = shape["label"]
            pre_label.append(label)# 이전 레이블 값 저장        
            counter = Counter(pre_label)    
            # 중복된 값을 포함하는 리스트 생성
            duplicates = [value for value, count in counter.items() if count > 1]
            if len(duplicates)>0: #중복된 요소 있을때
                for x in duplicates:
                    if x != label: #레이블이 중복된 값이 아닐때만 실행
                        if 'Breakage' in label:
                            text= imgpath+' 1 1 1\n'
                        elif 'Scratched' in label:
                            text= imgpath+' 2 2 1\n'
                        elif 'Crushed' in label:
                            text= imgpath+' 3 3 1\n'
                        elif 'Separated' in label:
                            text= imgpath+' 4 4 1\n'
                        # ...
                        with open(text_file_path, "a") as f:
                            f.write(text)   
            else: #중복된 요소 없을때
                        if 'Breakage' in label:
                            text= imgpath+' 1 1 1\n'
                        elif 'Scratched' in label:
                            text= imgpath+' 2 2 1\n'
                        elif 'Crushed' in label:
                            text= imgpath+' 3 3 1\n'
                        elif 'Separated' in label:
                            text= imgpath+' 4 4 1\n'
                        # ...
                        with open(text_file_path, "a") as f:
                            f.write(text)              

#훈련데이터 검증데이터 설정 7:3비율
def split_text_file(work_path, rate):
    input_file=os.path.join(work_path,'list.txt')
    output_file1=os.path.join(work_path,'trainval.txt')
    output_file2=os.path.join(work_path,'test.txt')
    with open(input_file, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    split_index = int(total_lines * rate)

    with open(output_file1, 'w') as file1, open(output_file2, 'w') as file2:
        for i, line in enumerate(lines):
            if i < split_index:
                file1.write(line)
            else:
                file2.write(line)

def create_folder(work_path):
    import shutil
    car_path =os.path.join(work_path,'car')
    os.makedirs(car_path, exist_ok=True)
    #annotations폴더 생성
    os.makedirs(os.path.join(car_path,'annotations'), exist_ok=True)
    annotation_path =os.path.join(car_path,'annotations')
    #trimap폴더 생성
    shutil.copytree(os.path.join(work_path,'trimaps'), os.path.join(annotation_path,'trimaps'))
    #이미지 폴더 생성 , 복사
    shutil.copytree(os.path.join(work_path,'images'), os.path.join(car_path,'images'))
    #xmls 폴더 생성
    shutil.copytree(os.path.join(work_path,'xmls'), os.path.join(annotation_path,'xmls'))
    #txt파일 복사
    shutil.copy(os.path.join(work_path,'list.txt'), os.path.join(annotation_path,'list.txt'))
    shutil.copy(os.path.join(work_path,'trainval.txt'), os.path.join(annotation_path,'trainval.txt'))
    shutil.copy(os.path.join(work_path,'test.txt'), os.path.join(annotation_path,'test.txt'))
    

def create_tar_gz(work_path):
    import tarfile
    car_path =os.path.join(work_path,'car')
    img_source_folder =os.path.join(car_path, 'images')
    img_output_file =os.path.join(car_path ,'images.tar.gz')
    with tarfile.open(img_output_file, "w:gz") as tar:
        tar.add(img_source_folder, os.path.basename(img_source_folder))  # 폴더를 루트로 추가
    ann_source_folder =os.path.join(car_path, 'annotations')
    ann_output_file =os.path.join(car_path ,'annotations.tar.gz')
    with tarfile.open(ann_output_file, "w:gz") as tar:
        tar.add(ann_source_folder, os.path.basename(ann_source_folder))  # 폴더를 루트로 추가    

