import requests
import urllib,os
from bs4 import BeautifulSoup
import tqdm
url_entrance = {
                "Accessory":["http://www.makehumancommunity.org/clothes.html?field_clothes_category_value=Accessory&field_clothes_status_value=All"
                           ,"http://www.makehumancommunity.org/clothes.html?field_clothes_category_value=Accessory&field_clothes_status_value=All&page=1"]
                }



def download_one_file(url,filepath):
    urllib.request.urlretrieve(url, filename=filepath)
    print(filepath)

def get_html_soup(url):
    webpage = requests.get(url)
    content = webpage.text
    soup = BeautifulSoup(content, 'html.parser')
    return soup

def download(url,category):
    soup = get_html_soup(url)
    tbody = soup.find("tbody")

    td = tbody.find_all('td')
    file_dirpath = 'assets/Clothes/' + category
    if not os.path.exists(file_dirpath):
        os.mkdir(file_dirpath)
    for t in td:
        if t.a:
            print(t.a.text)
            try:
                url_case = "http://www.makehumancommunity.org"+t.a['href']
                soup_case = get_html_soup(url_case)
                spans = soup_case.find_all("span",class_="file")
                asset_name = t.a.text
                asset_name = asset_name.replace(" ","_")

                for s in spans:
                    url_download = s.a['href']



                    file_dirpath = 'assets/Clothes/'+category+'/'+ asset_name
                    filepath = file_dirpath+"/"+ url_download.split('/')[-1]
                    if not os.path.exists(file_dirpath):
                        os.mkdir(file_dirpath)
                    download_one_file(url_download,filepath)
                fs = soup_case.find_all('figure')
                for f in fs:
                    url_texture = f.a['href']
                    filepath = file_dirpath + "/" + url_texture.split('/')[-1]
                    download_one_file(url_texture, filepath)
            except Exception as e:

                continue

if __name__=="__main__":
    for k,v in url_entrance.items():
        for url in v:
            download(url,k)