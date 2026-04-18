import json

data = json.load(open('data/bncc_embeddings.json', 'r', encoding='utf-8'))

historia9_keywords = [(k, v) for k, v in data.items() if 'História|9º' in k and 'keyword' in k]

print(f'Keywords História 9º: {len(historia9_keywords)}')
print('\nPrimeiros 20:')
for k, v in historia9_keywords[:20]:
    keyword = k.split('|')[-1]
    print(f'  {keyword}')

# Procurar especificamente por "varguista" ou "período"
print('\n\nProcurando "varguista":')
varguista = [k for k in data.keys() if 'varguista' in k.lower()]
print(f'  Encontrados: {len(varguista)}')
for k in varguista[:5]:
    print(f'    {k}')

print('\n\nProcurando "período":')
periodo = [k for k in data.keys() if 'keyword' in k and 'período' in k.lower()]
print(f'  Encontrados: {len(periodo)}')
for k in periodo[:10]:
    keyword = k.split('|')[-1]
    print(f'    {keyword}')
