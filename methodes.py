# Importer les packages
import pandas as pd
from requests import get
from bs4 import BeautifulSoup as bs
import time
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np

# Scraper les données par catégorie
def scraper_categorie(categorie, nb_pages=11):
    df_final = pd.DataFrame()
    # Liste des catégories à scraper
    categories = ['chiens', 'moutons', 'autres-animaux', 'poules-lapins-et-pigeons']
    print(f"Scraping de la catégorie: {categorie}")
    
    for ind_page in range(1, nb_pages + 1):
        url = f'https://sn.coinafrique.com/categorie/{categorie}?page={ind_page}'
        
        try:
            res = get(url)
            soup = bs(res.content, 'html.parser')
            containers = soup.find_all('div', 'col s6 m4 l3')
            
            data = []
            for container in containers:
                try:
                    Nom = container.find('p', 'ad__card-description').a.text
                    Prix = container.find('p', 'ad__card-price').text.strip('CFA')
                    Adresse = container.find('p', 'ad__card-location').span.text.strip()
                    Details = container.find('p', 'ad__card-description').a.get('title', 'N/A')
                    url_image = container.find('img')['src']
                    
                    if categorie == 'poules-lapins-et-pigeons':
                        dic = {
                            'Details': Details,
                            'Prix': Prix,
                            'Adresse': Adresse,
                            'url_image': url_image
                        }
                    else:
                        dic = {
                            'Nom': Nom,
                            'Prix': Prix,
                            'Adresse': Adresse,
                            'url_image': url_image
                        }
                    data.append(dic)
                except:
                    pass
            
            if data:
                data = pd.DataFrame(data)
                df_final = pd.concat([df_final, data], axis=0).reset_index(drop=True)
            
            print(f"  Page {ind_page}/{nb_pages} - {len(data)} annonces trouvées")
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"  Erreur sur la page {ind_page}: {e}")
            continue
    
    return pd.DataFrame(df_final)

# upputer les valeurs aberrantes (outliers) avec la méthode Winsorization
def impute_outliers_winsorization(data):
    for col in data.select_dtypes('number').columns:
        if not (-0.5 < data[col].skew() < 0.5):  # vérifier si la distribution est asymétrique
            lower_bound = np.percentile(data[col], 5)
            upper_bound = np.percentile(data[col], 95)
            data[col] = data[col].clip(lower_bound, upper_bound)

# imputer les valeurs aberrantes (outliers) avec la méthode IQR
def impute_outliers_iqr(data):
    for col in data.select_dtypes('number').columns:
        if -0.15 < data[col].skew() < 0.15 :  # vérifier si la distribution est symétrique
            Q1 = np.quantile(data[col], 0.25)
            Q2 = np.quantile(data[col], 0.5)
            Q3 = np.quantile(data[col], 0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            data[col] = np.where(data[col] < lower, lower, np.where(data[col] > upper, upper, data[col]))