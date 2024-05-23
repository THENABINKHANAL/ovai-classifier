from fathomnet.api import boundingboxes


concepts = boundingboxes.find_concepts()[1:]

f = open("species.csv", "w")
f.write('name,start_id\n')
for name in concepts:
    name = name.replace(',', '').replace('/', '').replace('.', '').replace('"', '').replace('(', '').replace(')', '')
    f.write(name+',0\n')
f.close()

