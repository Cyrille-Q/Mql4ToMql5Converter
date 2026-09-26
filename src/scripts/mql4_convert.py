import json

def convert_file(input_filepath: str, output_filepath: str):
    with open(input_filepath, 'r', encoding='utf-8') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line)
            text = data['text']
            # Remplacer chaque séquence \n par un saut de ligne
            converted_text = text.replace('\\n', '\n')
            # Écrire le texte converti suivi d'un saut de ligne
            outfile.write(converted_text + '\n')


convert_file("mql4_dataset.jsonl", "mql4_dataset.txt")