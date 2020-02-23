import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random
import cv2

digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
lower_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
upper_letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

char_list = open('char_std_5990.txt', 'r', encoding='utf-8').readlines()
char_list = [ch.strip('\n') for ch in char_list]

def prob(score):
    return not(random.randint(0, 100) > score)

def phone_number(length=10):
    select_range = digits + [' ', '+']
    phone = ""
    form = random.choice(select_range)
    for i in range(length):
        latter = random.choice(select_range)
        if prob(30):
            latter = form
        form = latter
        phone = phone + latter

    return phone

def identity_card(length=10):
    select_range = digits + ["X", 'x']
    id = ""
    for i in range(length):
        id = id + random.choice(select_range)
    return id

def mailbox(length = 10):
    select_range = digits + lower_letters + upper_letters +["@", "."]
    mail = ""
    for i in range(length):
        mail = mail + random.choice(select_range)
    return mail

def image_enhance(img,blur_porb=90, line_prob=20, bright_range=0.4, contrast_range=0.4):
    w,h=img.size
    # add line
    if prob(line_prob):
        if prob(50):
            rot_x = random.randint(1, w-1)
            ImageDraw.ImageDraw(img).line((rot_x, 0, rot_x, h), (0, 0, 0))
        if prob(50):
            rot_y = random.randint(1, h-1)
            ImageDraw.ImageDraw(img).line((0, rot_y,w, rot_y), (0, 0, 0))
    from PIL import ImageFilter, ImageEnhance
    img = ImageEnhance.Brightness(img).enhance(random.uniform(bright_range, 1))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(contrast_range, 1))
#    if prob(blur_porb):
#        img = img.filter(ImageFilter.GaussianBlur(radius=2))
    return img

def generate_ocr(input_width, input_height, expend_prob=90, expend_scale=0.8, enhance_prob=90):
    bg_color = (255, 255, 255)
    fg_color = (0, 0, 0)

    font_ch = ImageFont.truetype('simsun.ttf', input_height-1, 0)

    img = Image.new("RGB", (input_width, input_height), bg_color)
    string_len = 2*input_width // input_height
    string = random.choice([phone_number, identity_card, mailbox])(string_len)

    ImageDraw.Draw(img).text((0, 0), string, fg_color, font=font_ch)
    string = string.replace(" ", "")
    if prob(expend_prob):
        board_img = Image.new("RGB", (input_width, input_height), bg_color)
        scale = random.uniform(expend_scale, 0.99)
        rc_wd = int(scale*input_width)
        rc_hg = int(scale*input_height)
        img = img.resize((rc_wd, rc_hg))

        offset_y = random.randint(0, input_height - rc_hg)
        offset_x = random.randint(0, input_width - rc_wd)
        board_img.paste(img, (offset_x,offset_y,offset_x + rc_wd, offset_y+rc_hg))
        img = board_img

    if prob(enhance_prob):
        img = image_enhance(img)


    def rotate_image(img, rotate_range=3):
        img = img.convert('RGBA')
        ratate = random.randint(-rotate_range, rotate_range)
        rot = img.rotate(ratate)
        bg_ = Image.new('RGBA', rot.size, (255,) * 4)
        # bg_ = Image.new("RGBA", img.size, bg_color)
        img = Image.composite(rot, bg_, rot)
        img = img.convert("RGB")
        return img

    #img = rotate_image(img).convert('L')


    #img = np.array(img, 'f') / 255.0 - 0.5

#    x[i] = np.expand_dims(img, axis=2)

#    label_length[i] = len(char)
#    input_length[i] = imagesize[1] // 8

    label = [char_list.index(k) - 1 for k in string]
    return img, label

if __name__ == '__main__':
    img,label = generate_ocr(280, 28)
    label = [char_list[id+1] for id in label]
    cv2.imshow(str(label), np.array(img))
    print(label)
    cv2.waitKey(0)
