import os
from PIL import Image

found = False
basedir = 'wikimedia'
for fn in os.listdir(basedir):
    path = os.path.join(basedir, fn)
    for file in os.listdir(path):
        file = os.path.join(path, file)
        im = Image.open(file)
        color_count = im.getcolors()
        if color_count:
            try:
                os.remove(file)
            except:
                pass
    
    for index, file in enumerate(sorted(os.listdir(path), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))):
        try:
            os.rename(os.path.join(path, file), os.path.join(path, str(index)+'.jpg'))
        except:
            pass
    try:
        if '0.jpg' in os.listdir(path):
            os.rename(os.path.join(path, '0.jpg'), os.path.join(path, str(len(os.listdir(path)))+'.jpg'))
        if not os.listdir(path):
            os.removedirs(os.path.join(os.getcwd(),path))
        os.rename(path, os.path.join(basedir, fn.replace('_', ' '))) 
    except:
        pass
    
    
    
    
    
    