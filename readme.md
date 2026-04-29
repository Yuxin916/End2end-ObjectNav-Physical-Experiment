Environment Setup

```bash
cd ~/projects/VLN_CL_CoTNav/End2end-ObjectNav-Physical-Experiment
python3 -m venv .venv
source .venv/bin/activate

python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install transformers==4.37.2 deepspeed==0.14.4 accelerate==0.33.0 pyyaml

python -c "import torch; print(torch.__version__, torch.version.cuda)" 
# 2.5.1+cu121 12.1
python -m pip install -U wheel setuptools==79.0.1 ninja packaging cython
python -m pip install -U colcon-common-extensions pyyaml
python -m pip install catkin_pkg empy==3.3.4 lark ultralytics 
python -m pip install transformers==4.37.2 deepspeed==0.14.4 accelerate==0.33.0 timm==1.0.22 peft==0.10.0
python -m pip install \
  tensorrt-cu12==10.13.3.9 \
  tensorrt-cu12-bindings==10.13.3.9 \
  tensorrt-cu12-libs==10.13.3.9


python -m pip install numpy==1.26.4
MAX_JOBS=4 python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

start command:

```bash
conda deactivate
cd muyi/End2end-ObjectNav-Physical-Experiment/
source .venv/bin/activate
bash start_tmux_VLN_sim.sh
```

```bash
source .venv/bin/activate
bash start_tmux_VLN_env.sh
```

```bash
pkill -9 -f ros
pkill -9 -f unity
```