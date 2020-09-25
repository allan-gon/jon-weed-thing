import logging
import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field
import pickle

with open("app/api/vect", 'rb') as file:
    vect = pickle.load(file)
with open("app/api/nearest", 'rb') as file:
    nearest = pickle.load(file)

log = logging.getLogger(__name__)
router = APIRouter()

data = pd.read_csv('https://raw.githubusercontent.com/build-week-medcabinet-ch/data-science/master/data/final%20(1).csv')
ohe = pd.read_csv('https://raw.githubusercontent.com/build-week-medcabinet-ch/data-science/master/notebooks/psuedo_ohe.csv')

effect = {'Aroused', 'Creative', 'Dry Mouth',
          'Energetic', 'Euphoric', 'Focused',
          'Giggly', 'Happy', 'Hungry',
          'None', 'Relaxed', 'Sleepy',
          'Talkative', 'Tingly', 'Uplifted'}

flavor = {'Ammonia', 'Apple', 'Apricot', 'Berry',
          'Blue', 'Blueberry', 'Butter', 'Cheese',
          'Chemical', 'Chestnut', 'Citrus', 'Coffee',
          'Diesel', 'Earthy', 'Flowery', 'Fruit', 'Grape',
          'Grapefruit', 'Honey', 'Lavender', 'Lemon',
          'Lime', 'Mango', 'Menthol', 'Mint', 'Minty',
          'None', 'Nutty', 'Orange', 'Peach', 'Pear', 'Pepper',
          'Pine', 'Pineapple', 'Plum', 'Pungent', 'Rose', 'Sage',
          'Skunk', 'Spicy/Herbal', 'Strawberry', 'Sweet', 'Tar',
          'Tea', 'Tobacco', 'Tree', 'Tropical', 'Vanilla', 'Violet', 'Woody'}


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""
    Effects: str = Field(..., example='Creative,Energetic,Tingly,Euphoric,Relaxed')
    Type: str = Field(..., example='hybrid')
    Flavors: str = Field(..., example='Earthy,Sweet,Citrus')
    Description: str = Field(..., example="Helps with back pain.")

@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    types = {'hybrid': 0, 'sativa': 1, 'indica': 2}
    temp = pd.DataFrame(data=[[0] * 66], columns=['Pepper', 'Spicy/Herbal', 'Pine', 'Grapefruit', 'Apricot', 'Peach',
                                                  'Mint', 'Tea', 'Tar', 'Cheese', 'Vanilla', 'None', 'Minty', 'Diesel',
                                                  'Woody', 'Citrus', 'Sage', 'Ammonia', 'Fruit', 'Violet', 'Skunk',
                                                  'Butter', 'Flowery', 'Blueberry', 'Rose', 'Pineapple', 'Pear', 'Lime',
                                                  'Strawberry', 'Coffee', 'Berry', 'Sweet', 'Earthy', 'Nutty', 'Blue',
                                                  'Chemical', 'Pungent', 'Orange', 'Plum', 'Tropical', 'Apple',
                                                  'Tobacco',
                                                  'Honey', 'Chestnut', 'Mango', 'Menthol', 'Lemon', 'Tree', 'Grape',
                                                  'Lavender', 'Focused', 'Giggly', 'Sleepy', 'Uplifted', 'Relaxed',
                                                  'Happy', 'Talkative', 'Creative', 'Hungry', 'Energetic', 'None.1',
                                                  'Aroused', 'Euphoric', 'Dry Mouth', 'Tingly', 'Type'])
    for i in item.Effects.split(","):
        if i.capitalize() in effect:
            temp[i][0] = 1
    for i in item.Flavors.split(","):
        if i.capitalize() in flavor:
            temp[i][0] = 1
    temp['Type'] = types[item.Type.lower()]
    dtm = pd.DataFrame(np.array(vect.transform([item.Description]).todense()[0]))
    temp = pd.concat([temp, dtm], axis=1)
    neighbors = nearest.kneighbors([temp.iloc[0]])[1][0]
    return {
        "predictions": [ohe.iloc[i]['Strain'] for i in neighbors],
        "description": data[data['Strain'] == ohe.iloc[neighbors[0]]['Strain']].Description.values[0],
        "type": data[data['Strain'] == ohe.iloc[neighbors[0]]['Strain']].Type.values[0],
        "Effects": data[data['Strain'] == ohe.iloc[neighbors[0]]['Strain']].Effects.values[0],
        "Flavors": data[data['Strain'] == ohe.iloc[neighbors[0]]['Strain']].Flavors.values[0],
    }