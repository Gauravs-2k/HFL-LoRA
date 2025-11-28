import sys, json, os
path_file = os.path.join(os.getcwd(), 'path_debug.json')
with open(path_file, 'w') as f:
    json.dump({'sys_path': sys.path}, f, indent=2)
try:
    import sklearn
    print('sklearn import success, location:', sklearn.__file__)
except Exception as e:
    print('sklearn import error:', e)
