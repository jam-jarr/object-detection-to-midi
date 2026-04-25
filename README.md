# Object Detection to MIDI

## Installation

`python -m venv .venv`

`source .venv/bin/activate`

`pip install -r requirements.txt`

## Usage

`python main.py`

## TODO

- [x] Render input frame separately, scale label boxes to input frame size
- [x] JSON file for class mappings
  - Discrete array for choices
  - Mutually exclusive notes, `X detected > X<=len(array)` notes played
  - If max notes are reached, increase velocity of notes (ADD)
    - Configuration parameter
- [ ] Different ports for different classes
- [ ] Map confidence to velocity
- [ ] Switch to UV

## IDEAS

- Procedural audio generation?
  - Manhattan
- Fine tuning?
  - Leaf
  - Sparkle
