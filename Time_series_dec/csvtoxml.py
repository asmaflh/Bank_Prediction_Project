

import xml.etree.ElementTree as ET
import pandas as pd
# Read the CSV file
df = pd.read_csv('Csvs/All_context.csv')
# Get the unique weeks from the Week_Number column
weeks = df['Week_Number'].unique()
# Define the attributes
attrs = df.columns[df.columns != 'Week_Number']
# Create the root element of the XML
root = ET.Element('Galicia_Document')
# Create the BinaryContext element
binary_context = ET.SubElement(root, 'BinaryContext')
binary_context.set('numberObj', str(len(weeks)))
binary_context.set('numberAtt', str(len(attrs)))
# Add the Name element
name = ET.SubElement(binary_context, 'Name')
name.text = 'Ctx_10'
# Add Object elements
for week in weeks:
    obj = ET.SubElement(binary_context, 'Object')
    obj.text = str(week)
# Add Attribute elements
for attr in attrs:
    att = ET.SubElement(binary_context, 'Attribute')
    att.text = attr
# Add Object elements
for idxO, row in df.iterrows():
    for idxA, value in enumerate(row[1:], start=0):
        if value == 1:
            bin_rel = ET.SubElement(binary_context, 'BinRel')
            bin_rel.set('idxO', str(idxO))  # Row number
            bin_rel.set('idxA', str(idxA))
# Create an ElementTree object from the root element
tree = ET.ElementTree(root)
# Write the ElementTree object to an XML file
tree.write('Xmls/All_context.bin.xml')
