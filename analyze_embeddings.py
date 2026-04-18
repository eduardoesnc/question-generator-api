import json

data = json.load(open('data/bncc_embeddings.json', 'r', encoding='utf-8'))

print(f'Total embeddings: {len(data)}')

tipos = {}
for k, v in data.items():
    t = v.get('tipo', 'unknown')
    tipos[t] = tipos.get(t, 0) + 1

print(f'\nPor tipo:')
for k, v in sorted(tipos.items()):
    print(f'  {k}: {v}')

print(f'\nExemplo de chave:')
exemplo_key = list(data.keys())[0]
print(f'  {exemplo_key}')

print(f'\nExemplo de dados:')
exemplo_data = list(data.values())[0]
print(f'  Keys: {list(exemplo_data.keys())}')
print(f'  Texto: {exemplo_data.get("texto", "N/A")[:80]}...')
print(f'  Disciplina: {exemplo_data.get("disciplina", "N/A")}')
print(f'  Ano: {exemplo_data.get("ano", "N/A")}')

# Procurar por "vargas"
print(f'\n\nBuscando "vargas" nos embeddings:')
count = 0
for k, v in data.items():
    texto = v.get('texto', '').lower() if v.get('texto') else ''
    objeto = v.get('objeto', '').lower() if v.get('objeto') else ''
    if 'vargas' in texto or 'vargas' in objeto:
        print(f'\n  Key: {k}')
        print(f'  Texto: {v.get("texto", "N/A")[:100]}...')
        print(f'  Disciplina: {v.get("disciplina")} - Ano: {v.get("ano")}')
        count += 1
        if count >= 5:
            break

if count == 0:
    print('  Nenhum embedding encontrado com "vargas"!')

# Verificar História 9º
print(f'\n\nEmbeddings de História 9º:')
count = 0
for k, v in data.items():
    if v.get('disciplina') == 'História' and v.get('ano') == '9º':
        print(f'\n  Key: {k}')
        print(f'  Texto: {v.get("texto", "N/A")[:100]}...')
        print(f'  Tipo: {v.get("tipo")}')
        count += 1
        if count >= 10:
            break
