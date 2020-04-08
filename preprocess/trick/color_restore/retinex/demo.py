from mycode.plot import contrast_plot,demo_viper_with_retinex
from mycode.retinex import retinex_FM,retinex_SSR,retinex_MSR,retinex_MSRCR,retinex_gimp,retinex_MSRCP,retinex_AMSR
from mycode.tools import cv2_heq
import os.path
import cv2
import time

#img_dir=os.path.normpath(os.path.join(os.path.dirname(__file__),'./imgs'))
#img=cv2.imread(os.path.join(img_dir,'demo2.png'))

#img_path = "C:/Users/62349/Desktop/untitled2/water/000042.jpg"
#img_path = "C:/Users/62349/Desktop/untitled2/water/000001.jpg"
src_path = '/media/clwclw/data/2020water/image'
dst_path = '/media/clwclw/data/2020water/image_restore'
if not os.path.exists(dst_path):
    os.mkdir(dst_path)
# img_path = "./water/000001.jpg"
img_names = os.listdir(src_path)
img_names = [name for name in img_names if name.endswith('.jpg')]

t0 = time.time()
for idx, img_name in enumerate(img_names):
    img_path = os.path.join(src_path, img_name)
    img = cv2.imread(img_path)

    #test1
    #demo_viper_with_retinex(os.path.join(img_dir,'VIPeR.v1.0'),retinex_AMSR)

    #test2
    #contrast_plot(img,[retinex_SSR(img,15),retinex_SSR(img,80),retinex_SSR(img,250),retinex_MSR(img),retinex_gimp(img),retinex_MSRCR(img),retinex_MSRCP(img),retinex_FM(img),cv2_heq(img),retinex_AMSR(img,)],['src','SSR(15)','SSR(80)','SSR(250)','MSR(15,80,250,0.333)','Gimp','MSRCR','MSRCP','FM','cv2 heq','Auto MSR'],save=False)

    # clw test
    # start = time.time()
    # img_out = cv2_heq(img)
    # print('time use:%.3fs' % (time.time() - start))
    # cv2.imwrite('cv2_heq.jpg', img_out)

    start = time.time()
    img_out = retinex_AMSR(img)
    print('image index %d, time use:%.3fs' % (idx, time.time() - start))
    cv2.imwrite(os.path.join(dst_path, img_name), img_out)

print('--- total time use:%.3fs ---' % (time.time() - t0))