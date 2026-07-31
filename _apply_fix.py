import sys, os
sys.stdout.reconfigure(encoding='utf-8')

path = os.path.join(os.getcwd(), 'src', 'my_agent', 'core', 'engine.py')
with open(path, encoding='utf-8') as f:
    lines = f.readlines()

# Fix line 254 (0-indexed: 253): hook_data -> _hdata
if '_execute_tool' in lines[252]:
    for i in range(250, len(lines)):
        if i < len(lines) and 'hook_data' in lines[i] and '_execute_tool' not in lines[i]:
            # This is inside _execute_tool
            old = lines[i]
            new = old.replace('hook_data', '_hdata')
            print(f"  FIX line {i+1}: {old.rstrip()} -> {new.rstrip()}")
            lines[i] = new

# Fix line 237 (0-indexed: 236): context is not None -> _context is not None  
# Actually the broader issue: `if context is not None` references an undefined 'context' variable
# Since this is inside _execute_tool which doesn't receive context param, remove that block
for i in range(235, 242):
    if i < len(lines) and 'context is not None' in lines[i]:
        old = lines[i]
        new = lines[i].replace('if context is not None:', 'if True:')  # Keep context forwarding via _hdata
        print(f"  FIX line {i+1}: {old.rstrip()} -> {new.rstrip()}")
        lines[i] = new

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("DONE")
