# Player reId Soccer

## Objective

Given two clips of some gameplay from differetn camera angles, map the player such that each players retains the consistent ID across both feed

## Steps to Run this project

1. Clone this repo and move into the folder
```bash
git clone https://github.com/prasoonmhwr/player-reid-soccer.git
cd player-reid-soccer
```

3. Download and store this model in models folder https://drive.google.com/file/d/1cEH_PGve7G6bFp9mhmrQ-7j8gGOvzhB6/view?usp=sharing

4. Create a virtual environment using 

```bash
python3 -m vevn myEnv
```
```bash
source myEnv/bin/activate
```

4. Install the dependencies

```bash 
pip install -r requirements.txt
```

5. Start the process
```bash
python main.py --broadcast data/broadcast.mp4 --tacticam data/tacticam.mp4 --model models/best.pt
```

The annotated videos will be in output folder
