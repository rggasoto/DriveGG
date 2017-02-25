from xml.etree import ElementTree
import cv2
import os
import csv

def getSteeringFromXml(name):
    xmlFile = ElementTree.parse("{}.xml".format(name))
    xmlRoot = xmlFile.getroot()
    steering = xmlRoot.find("Steering").text
    return steering
#Get name of xml files removing file type ".xml"
xmlfiles = [i[:-4] for i in os.listdir() if i.endswith('xml')]
csvFile = csv.writer(open("training.csv",'a',newline=''))
for i in xmlfiles:
    row = ["{}\{}.png".format(os.path.abspath(''),i),'','',getSteeringFromXml(i)]
    csvFile.writerow(row)
    
