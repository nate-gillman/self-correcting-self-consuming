### Rendering Video

Getting off-screen rendering to work requires the following steps. After completing the uhc v3 environment steps, write this environment variable:
```
export PYOPENGL_PLATFORM=osmesa
```

Load this module if not already:
```
module load mesa
module load glew
```

Install these:
```
pip install PyOpenGL -U
pip install --force-reinstall imageio==2.23.0
pip install numpy==1.23.1
```
