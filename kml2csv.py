# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 10:48:29 2025

@author: sletizia
"""
import os
cd=os.getcwd()
import xml.etree.ElementTree as ET
import csv
import sys

# Usage: python extract_kml_points.py doc.kml output.csv
if len(sys.argv)==1:
    kml_file = os.path.join(cd,'data/doc.kml')
    csv_file = os.path.join(cd,'data/doc.csv')
else:
    kml_file = sys.argv[1]
    csv_file = sys.argv[2]

tree = ET.parse(kml_file)
root = tree.getroot()

# KML uses namespaces, but we just use wildcard {*} to ignore them
placemarks = root.findall('.//{*}Placemark')

rows = []

for pm in placemarks:
    name_el = pm.find('{*}name')
    name = name_el.text if name_el is not None else ""

    point = pm.find('.//{*}Point')
    if point is None:
        continue  # skip if not a point placemark

    coords_el = point.find('{*}coordinates')
    if coords_el is None or not coords_el.text.strip():
        continue

    coords = coords_el.text.strip().split(',')  # lon,lat,alt
    lon = coords[0]
    lat = coords[1]

    rows.append([name, lat, lon])

# Write CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "lat", "lon"])
    writer.writerows(rows)

print(f"Extracted {len(rows)} point placemarks to {csv_file}")
