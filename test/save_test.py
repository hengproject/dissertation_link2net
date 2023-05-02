import os.path

from util.save_util import *

hi = {
    "hi": 123,
    6: "hi"
}
hi2 = {
    "shit": True
}

path = r'C:\DATA\projects\dissertation\save_data\1.pth'
path = os.path.join(path)
save_model(path, arg=hi, arg2=hi2)


