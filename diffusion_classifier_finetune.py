import os

concepts_list = []

for folder in os.listdir('train40'):
  words = folder.split()
  print(words)
  prompt = words[0] + ' to the ' + words[1] + ' of a ' + words[2]
  concepts_list.append(
    {
        "instance_prompt":      "photo of a " + prompt,
        "instance_data_dir":    "train40/" + folder,
        "class_prompt":         "",
        "class_data_dir":       ""
    })

import json
import os
for c in concepts_list:
    os.makedirs(c["instance_data_dir"], exist_ok=True)

with open("concepts_list.json", "w") as f:
    json.dump(concepts_list, f, indent=4)