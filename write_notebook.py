import json, os, io, re

def read(path):
    with io.open(path, 'r', encoding='utf-8') as f:
        return f.read()

def code_cell(src):
    return {"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(True)}

def md_cell(text):
    return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}

nb = {"nbformat":4,"nbformat_minor":5,
      "metadata":{"colab":{"name":"ReConPatch_Colab_Full.ipynb"},
                  "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                  "language_info":{"name":"python"}},
      "cells":[]}

nb['cells'].append(md_cell('# ReConPatch Colab (Full Source Inline)\nThis notebook inlines the code from `src/` and `main.py` into separate cells, with minimal compatibility glue for Colab.\n- Code blocks mirror your files; small additions register them as `src.*` modules for imports.\n- A compatibility cell ensures mixed precision loss scaling works even if Colab TF lacks `get_scaled_loss()`.\n'))
setup = '''#@title Setup: Install dependencies
import sys, subprocess

def pip_install(pkgs):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + pkgs)

# Ensure faiss-cpu and deps
try:
    import faiss  # noqa: F401
except Exception:
    pip_install(['faiss-cpu'])

pip_install(['tqdm', 'scikit-learn', 'pillow', 'matplotlib', 'scipy'])
import tensorflow as tf; import faiss, numpy as np; print('TF', tf.__version__)
'''
nb['cells'].append(code_cell(setup))
compat = '''#@title Compatibility: Mixed Precision LossScaleOptimizer shim (if missing)
import tensorflow as tf
mp = tf.keras.mixed_precision
if not hasattr(mp, 'LossScaleOptimizer'):
    class _CompatLSO:
        def __init__(self, opt): self._opt = opt
        def get_scaled_loss(self, loss): return loss
        def get_unscaled_gradients(self, grads): return grads
        def apply_gradients(self, gv): return self._opt.apply_gradients(gv)
        def __getattr__(self, n): return getattr(self._opt, n)
    mp.LossScaleOptimizer = _CompatLSO
print('LossScaleOptimizer:', mp.LossScaleOptimizer)
'''
nb['cells'].append(code_cell(compat))

modules = [
 ('src/backbone.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.backbone\'); _m.backbone_model = backbone_model; sys.modules[\'src.backbone\'] = _m\n'),
 ('src/projector.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.projector\'); _m.Projector = Projector; sys.modules[\'src.projector\'] = _m\n'),
 ('src/patch_aggregator.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.patch_aggregator\'); _m.extract_patches = extract_patches; _m.aggregate = aggregate; sys.modules[\'src.patch_aggregator\'] = _m\n'),
 ('src/EMAnetwork.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.EMAnetwork\'); _m.EMANetwork = EMANetwork; sys.modules[\'src.EMAnetwork\'] = _m\n'),
 ('src/data_loader.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.data_loader\'); _m.collect_image_paths = collect_image_paths; _m.get_dataset = get_dataset; sys.modules[\'src.data_loader\'] = _m\n'),
 ('src/similarity_loss.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.similarity_loss\'); _m.compute_similarity = compute_similarity; _m.relaxed_contrastive_loss = relaxed_contrastive_loss; sys.modules[\'src.similarity_loss\'] = _m\n'),
 ('src/memorybank.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.memorybank\'); _m.MemoryBank = MemoryBank; sys.modules[\'src.memorybank\'] = _m\n'),
 ('src/evaluate.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.evaluate\'); _m.evaluate = evaluate; sys.modules[\'src.evaluate\'] = _m\n'),
 ('src/trainer.py', 'import sys, types as _types\n_m = _types.ModuleType(\'src.trainer\'); _m.train = train; sys.modules[\'src.trainer\'] = _m\n'),
]

for path, reg in modules:
    nb['cells'].append(md_cell(f'## {path}'))
    src = read(path)
    cell_src = src + ('' if reg is None else '\n\n# Register as src module for imports\n' + reg)
    nb['cells'].append(code_cell(cell_src))

main_src = read('main.py')
main_src = re.sub(r"ROOT_DIR\s*=.*", "ROOT_DIR = '/content/mvtec'  #@param {type:'string'}", main_src)
main_src = re.sub(r"epochs=\s*10", "epochs=1", main_src)
nb['cells'].append(md_cell('## main.py (parameterized for Colab path)'))
nb['cells'].append(code_cell(main_src))

out = 'ReConPatch_Colab_Full.ipynb'
with io.open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False)
print('Wrote', out, 'with', len(nb['cells']), 'cells')
