#-*- coding:utf-8 -*-
import os
import cv2
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
image_files = glob('./test_images/*.*')

def progress_single(img):
    result, image_framed, text_recs= ocr.model(img)
    h,w,_= img.shape
    for key in result:
        res = result[key]
        pt1 = res[0]
        pt2 = res[1]
        img = cv2.rectangle(img, pt1=(int(pt1[0]*w),int(pt1[1]*h)), pt2=(int(pt2[0]*w), int(pt2[1]*h)), color=(0, 255,0))
        #cv2.putText(img, result[key][2], (int(pt1[0]*w),int(pt1[1]*h)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0))
        result[key][0]=(int(pt1[0]*w),int(pt1[1]*h))
        result[key][1]=(int(pt2[0]*w), int(pt2[1]*h))
        print(result[key][2])
    cv2.imshow("demo", img)
    cv2.waitKey(0)
    return result

def write_word_to_json(result, offset_add=0):
    pt1 = result[0]
    pt2 = result[1]
    json_str = {}
    json_str.update({"words": result[2]})
    json_str.update({"location": {
        "width": (pt2[0]-pt1[0]),
        "top": (pt1[1]+offset_add),
        "left": pt2[0],
        "height": (pt2[1]-pt1[1])
    }})
    return json_str

def progress(img):
    h,w,_ = img.shape
    n = 4
    distance = h // n
    res = {}
    words_result = {"words_result": []}
    top_num = 0
    for i in range(n):
        sub_add = i*distance
        image = img[i*distance:min((i+1)*distance,h), :,:]
        result = progress_single(image)
        top_num = len(result) + top_num
        for j in result:
            #print(result)
            #print(j)
            words_result["words_result"].append(write_word_to_json(result[j], sub_add))
    res.update(words_result)
    res.update({"words_result_num": top_num})
    res.update({"h": h, "w": w})
    return res

if __name__ == '__main__':

    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        res = progress(image)
        print(res)
        exit(0)
        t = time.time()
        h,w,_ = image.shape
        image_top = image[0:h//2,:,:]
        image_bottom = image[h//2, :, :]
        #json_str = {}
        #json_str.update({"":})

        # detecte top
        progress(image_top)
        print(image_top.shape)
        result, image_framed, text_recs= ocr.model(image_top)
        h,w,_= image_top.shape
        for key in result:
            res = result[key]
            pt1 = res[0]
            pt2 = res[1]
            image_top = cv2.rectangle(image_top, pt1=(int(pt1[0]*w),int(pt1[1]*h)), pt2=(int(pt2[0]*w), int(pt2[1]*h)), color=(0, 255,0))
            print(result[key][1])

        print(image_framed.shape)
        cv2.imshow("demo", image_top)
        cv2.waitKey(0)
        exit(0)


        result, image_framed, text_recs= ocr.model(image)



        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(key)
            print(result[key][1])

