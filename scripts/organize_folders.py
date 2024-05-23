import os

found = False
basedir = 'inaturalist'
for fn in os.listdir(basedir):
    path = os.path.join(basedir, fn)
    if not os.path.isdir(path): 
        continue
    if not os.listdir(path):
        os.removedirs(os.path.join(os.getcwd(),path))
        continue
    os.rename(path, os.path.join(basedir, fn.replace('_', ' ')))
    
    """
    for index, file in enumerate(os.listdir(path)):
        #print(index, file)
        if fn == 'Chorilia':
            found = True
        if not found:
            continue
        os.rename(os.path.join(path, file), os.path.join(path, str(index+1)+'.jpg'))
    """
    
    
    
    
    