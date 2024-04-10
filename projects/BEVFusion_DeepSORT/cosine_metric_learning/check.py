import json

f = open('object_data.json')
instances = json.load(f)

total = 0
object_dict = {}
for token in instances:
    if len(instances[token]['images']) >= 5:
        total += 1
        object_dict[token] = instances[token]
        
with open('object_data_resumed.json', 'w') as f:
    json.dump(object_dict, f, indent=4)
        

        
print(f'Total: {total}')

f.close()