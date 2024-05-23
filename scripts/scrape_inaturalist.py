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
    if(os.path.isdir('inaturalist/'+name+'/') == False):
        os.mkdir('inaturalist/'+name+'/')
            
    link = 'https://www.inaturalist.org/observations?place_id=any&taxon_name='+name.replace(' ', '%20')
    #link = input("Input Your Link: ")
    #print(link)
    
    req = requests.get(link)
    soup = BeautifulSoup(req.text,'lxml')

    print(soup)
    soup = soup.find("div", {"id": 'result-grid'})
    print(soup)

    imgs=soup.find_all('div', {"class": 'thumbnail'})
    print(imgs)
    
    k = 1
    for i in imgs:
        #print(i)
        style =i['style']
        regex="(https://[^\s]+(jpeg|jpg|png))"

        matches = re.findall(regex, style)
        print(matches)
        url = matches[0]
        
        print('Image Link:',k)
        print(url)
        
        response = requests.get(url,stream=True)
        img = Image.open(response.raw)
        
        w, h = img.size
        if w < 100 and h < 100:
            continue
        
        #plt.imshow(img)
        #plt.close()
        
        
                  
        img.save('inaturalist/'+name+'/{}.jpg'.format(str(k)))
        k+=1
        
        
        if k > 50:
            break
            

with open('completed-inaturalist.json') as f:
    completed = json.load(f)

concepts = boundingboxes.find_concepts()[1:]
for name in concepts:
    if name in completed:
        continue

    name = name.replace(',', '').replace('/', '').replace('.', '')
    print(name)
   
    scrape(name)
    
    completed.append(name)
    
    with open("completed-inaturalist.json", "w") as outfile:
        json.dump(completed, outfile)

    break

