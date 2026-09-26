import glob
import subprocess
import sys

f = sorted(glob.glob('/home/lf/level4/dataset/test/input/*.jpg'))[0]
print('测试图片:', f)
subprocess.run([sys.executable, 'test_one.py', '--image', f])
