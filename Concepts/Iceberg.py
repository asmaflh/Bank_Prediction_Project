import xml.etree.ElementTree as ET
import pandas as pd
# Parse the XML file
tree = ET.parse('Yb&Tw_FRcontext.lat.xml')
root = tree.getroot()
lattice = root.find("Lattice")
nb_concept = lattice.get("numberCpt")
conceptId_ref = []
lattice = []
# Find the concept with ID equal to 11
for concept in root.iter('Concept'):
    concept_id = concept.find('ID')
    if concept_id is not None and concept_id.text.strip() == nb_concept:
        # Get the UpperCovers element of the concept
        uppercovers = concept.find('UpperCovers')
        if uppercovers is not None:
            concept_refs = uppercovers.findall('Concept_Ref')
            for concept_ref in concept_refs:
                print(concept_ref.text)
                conceptId_ref.append(concept_ref.text)
        else:
            print(f"UpperCovers element not found for concept with ID {nb_concept}")
        break
else:
    print(f"Concept with ID {nb_concept} not found")
for cp in conceptId_ref:
    for concept in root.iter('Concept'):
        concept_id = concept.find('ID')
        if concept_id is not None and concept_id.text.strip() == cp:
            week = []
            prop = []
            # Get the UpperCovers element of the concept
            extent = concept.find('Extent')
            intent = concept.find('Intent')
            if extent is not None:
                object_refs = extent.findall('Object_Ref')
                for object_ref in object_refs:
                    week.append(object_ref.text)
            else:
                print(f"object element not found for concept with ID {nb_concept}")
            if intent is not None:
                attr_refs = intent.findall('Attribute_Ref')
                for attr_ref in attr_refs:
                    prop.append(attr_ref.text)
            else:
                print(f"attribute element not found for concept with ID {nb_concept}")
            item = {
                "Id":cp,
                "Extent": week,
                "Intent": prop
            }
            lattice.append(item)
            break
    else:
        print(f"Concept with ID {nb_concept} not found")
for lat in lattice:
    lat["Extent"].sort()
    print(lat["Intent"])

df = pd.DataFrame(lattice)
print(df)
print(df[df["Id"] == "493"]["Intent"])
