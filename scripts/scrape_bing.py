from bing_image_downloader import downloader
import requests


url = 'https://www.bing.com/images/async?q=Odostomia+japonica&first=1&count=20&adlt=on&qft=+filterui:color2-color+filterui:photo-photo'
response = requests.get(url)
with open("t.html", "w", encoding='utf8') as outfile:
    outfile.write(response.text)

#downloader.download('Odostomia japonica', limit=20,  output_dir='bing', adult_filter_off=False, force_replace=True, timeout=10, verbose=True, filter='+filterui:color2-color+filterui:photo-photo')
