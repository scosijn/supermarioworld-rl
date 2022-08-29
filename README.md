# Super Mario World AI
## Installation
### Prerequisites
- Python 3.8
- (Optional) `ffmpeg` or `avconv` to record videos
### Install using pip
Clone the repository or download the ZIP.
```
git clone https://github.com/scosijn/supermarioworld-rl
```
Navigate to the project directory:
```
cd supermarioworld-rl
```
Create a new virtual environment:
```
python -m venv .env
```
Activate the virtual environment:
#### Windows:
```
.\.env\Scripts\activate
```
#### Linux/macOS:
```
source .env/bin/activate
```
Install the required packages:
```
pip install -r requirements.txt
```
Import the ROM of the game:
```
python -m retro.import ROM
```
If everything worked correctly you should see the following message.
```
Importing SuperMarioWorld-Snes
Imported 1 games
```