import sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = os.path.join(os.getcwd(), 'src', 'my_agent', 'core', 'engine.py')
with open(path, encoding='utf-8') as f:
    src = f.read()

before = 'data={**hook_data,'
after = 'data={**_hdata,'

if before in src and after not in src.replace(before, after):
    new_src = src.replace(before, after)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print('FIXED: replaced hook_data -> _hdata in TOOL_ERROR path')
else:
    print('NOT NEEDED or already applied')
