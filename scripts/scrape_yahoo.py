# WEB Page Image Scraper
# https://github.com/agaraman0/Web_Page_Image_Scrapper/blob/master/web_page_scrapper.py

import requests
from bs4 import *
from PIL import Image
import matplotlib.pyplot as plt
import os
import json
from fathomnet.api import boundingboxes


def scrape(name):
    if(os.path.isdir('yahoo/'+name+'/') == False):
        os.mkdir('yahoo/'+name+'/')
            
    link = 'https://images.search.yahoo.com/search/images;_ylt=AwrFG4swED1mW4sR93GJzbkF?p='+name.replace(' ', '%20')+'&imgc=color&imgty=photo&ei=UTF-8&imgf=nonportrait&fr2=p%3As%2Cv%3Ai' 
    #link = input("Input Your Link: ")
    #print(link)
    
    req = requests.get(link)
    soup = BeautifulSoup(req.text,'lxml')

    soup = soup.find("div", {"id": 'res-cont'})

    imgs=soup.find_all('img')
    k = 1
    for i in imgs:
        #print(i)
        try:
            url =i['src']
        except:
            continue
        #print('Image Link:',k)
        #print(url)
        response = requests.get(url,stream=True)
        img = Image.open(response.raw)
        
        w, h = img.size
        if w < 100 and h < 100:
            continue
        
        #plt.imshow(img)
        #plt.close()
        
        
                  
        img.save('yahoo/'+name+'/{}.jpg'.format(str(k)))
        k+=1
        
        
        if k > 50:
            break
            

with open('completed.json') as f:
    completed = json.load(f)

concepts = boundingboxes.find_concepts()[1:]
for name in concepts:
    if name in completed:
        continue
    print(name)
    completed.append(name)
    name = name.replace(',', '').replace('/', '').replace('.', '')

    try:
        scrape(name)
    except:
        print('---- err')

    with open("completed.json", "w") as outfile:
        json.dump(completed, outfile)
            
    #break

