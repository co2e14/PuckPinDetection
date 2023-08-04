import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text))
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

annotation_path = '/Users/vwg85559/PuckPinDetection/images' # Change to your path
xml_df = xml_to_csv(annotation_path)
xml_df.to_csv('labels.csv', index=False)



dataframe = pd.read_csv('labels.csv')
train_data, val_data = train_test_split(dataframe, test_size=0.2, random_state=42)

train_data.to_csv('train_labels.csv', index=False)
val_data.to_csv('val_labels.csv', index=False)
