import os
from PIL import Image


class Cut_Picture():
    pic_dir = "D:\Link\Photos\手写矩形素材\新的"
    cut_h = 100
    cut_w = 100
    # new_h = 64
    # new_w = 64
    pic_sqn = "triangle_new5"

    def __init__(self):
        self.image = Image.open(self.pic_dir + "\\" + self.pic_sqn + ".jpg")

        self.pic_h, self.pic_w = self.image.size
        print(self.pic_h, self.pic_w)
        self.cut_pic()

    def cut_pic(self):

        folder = os.path.exists(self.pic_dir + "\\" + self.pic_sqn)

        if not folder:
            os.makedirs(self.pic_dir + "\\" + self.pic_sqn)

        for x in range(int(self.pic_w / self.cut_w)):
            for y in range(int(self.pic_h / self.cut_h)):
                box = (x * self.cut_w, y * self.cut_h, (x + 1) * self.cut_w, (y + 1) * self.cut_h)
                img = self.image.crop(box)
                img.save(self.pic_dir + "\\" + self.pic_sqn + "\\" + str(x) + str(y) + ".jpg")
                # img.resize((self.new_w, self.new_h), Image.ANTIALIAS).save(self.pic_dir + "\\" + self.pic_sqn + "\\" + str(x) + str(y) + ".jpg")


if __name__ == '__main__':
    a = Cut_Picture()
