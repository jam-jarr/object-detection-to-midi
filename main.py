import argparse
import json
from collections import defaultdict

import cv2
import mido
import numpy as np
import torch
from ultralytics import YOLO

VELOCITY_BASE = 80
MIDI_CHANNEL = 0
VELOCITY_ADD = 20

DEFAULT_MIDI = {
    "notes": [60],
    "color": (0, 255, 0),
}


class MidiController:
    def __init__(self, port_name="YOLOv5 Detection"):
        self.port = None
        self.port_name = port_name
        self.use_file_mode = False
        self.midi_file = None
        self.track = None
        self.active_notes = set()

    def open_port(self, port=None):
        available_ports = mido.get_output_names()
        print(f"Available MIDI outputs: {available_ports}")

        if port:
            try:
                self.port = mido.open_output(port)
                print(f"Connected to MIDI port: {port}")
                return True
            except Exception as e:
                print(f"Could not open port: {e}")

        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            print(f"Virtual MIDI port created: {self.port_name}")
            return True
        except Exception as e:
            print(f"Could not create virtual port: {e}")

        if available_ports:
            try:
                self.port = mido.open_output(available_ports[0])
                print(f"Connected to first available MIDI port: {available_ports[0]}")
                return True
            except Exception as e:
                print(f"Could not open first available port: {e}")

        return True

    def send_note_on(self, note, velocity, channel=MIDI_CHANNEL):
        if self.use_file_mode and self.track:
            self.track.append(
                mido.Message(
                    "note_on",
                    note=note,
                    velocity=velocity,
                    channel=channel,
                    time=0,
                )
            )
        elif self.port:
            self.port.send(
                mido.Message("note_on", note=note, velocity=velocity, channel=channel)
            )
        self.active_notes.add(note)

    def send_note_off(self, note, velocity=0, channel=MIDI_CHANNEL):
        if self.use_file_mode and self.track:
            self.track.append(
                mido.Message(
                    "note_off",
                    note=note,
                    velocity=velocity,
                    channel=channel,
                    time=0,
                )
            )
        elif self.port:
            self.port.send(
                mido.Message("note_off", note=note, velocity=velocity, channel=channel)
            )
        self.active_notes.remove(note)

    def close(self):
        for note in list(self.active_notes):
            self.send_note_off(note)

        if self.use_file_mode and self.midi_file:
            if self.track:
                self.track.append(mido.Message("end_of_track"))
                self.midi_file.save("detections.mid")
            print("MIDI file saved: detections.mid")
        elif self.port:
            self.port.close()
            print("MIDI port closed")


def get_note_list(class_name):
    class_conf = configuration.get(class_name, DEFAULT_MIDI)
    return class_conf.get("notes")


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--port",
    type=str,
    default=None,
    help="MIDI port name or virtual port name",
)

parser.add_argument(
    "--model",
    type=str,
    default="yolov5s",
    help="YOLOv5 model name",
)

parser.add_argument(
    "--conf",
    type=float,
    default=0.30,
    help="YOLOv5 confidence threshold",
)

parser.add_argument(
    "--iou",
    type=float,
    default=0.45,
    help="YOLOv5 IOU threshold",
)

parser.add_argument(
    "--debug",
    action="store_true",
    help="Debug mode",
)

parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Device index of video capture",
)


class args:
    pass


parser.parse_args(namespace=args)

if args.debug:
    print("Arguments and defaults:")
    print(vars(args))
    print("-" * 50)

# load configuration file
with open("configuration.json") as f:
    configuration = json.load(f)

if args.debug:
    print(configuration)
    print(configuration.get("person"))


# Load model
model = torch.hub.load(
    "ultralytics/yolov5", args.model, pretrained=True, _verbose=False
)

# model = YOLO("yolov8s.pt")

model.conf = args.conf
model.iou = args.iou

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize webcam
cap = cv2.VideoCapture(args.device)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize MIDI
midi = MidiController("YOLOv5 Detection")
midi.open_port(args.port)

print("Webcam opened. Press 'q' to quit.")

cv2.namedWindow("YOLOv5 Detections", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

available_classes = set()

for class_name, class_conf in configuration.items():
    if not class_conf.get("disabled", False):
        available_classes.add(class_name)

playing_notes = defaultdict(dict)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        original_frame = frame.copy()
        results = model(frame)

        if args.debug:
            print(f"Original frame size: {original_frame.shape}")
            print(f"Frame size: {frame.shape}")

        detections = results.pandas().xyxy[0]

        # Remove detections that are not in the available classes from the pandas dataframe
        detections_names = set(detections["name"])
        available_detections = detections_names & available_classes
        detections = detections[detections["name"].isin(available_detections)]

        current_detections = defaultdict(int)

        for _, row in detections.iterrows():
            class_name = row["name"]
            confidence = row["confidence"]

            class_conf = configuration.get(class_name, DEFAULT_MIDI)

            if args.debug:
                print(f"Class: {class_name}")
                print(f"Conf: {class_conf}")

            x1, y1, x2, y2 = (
                int(row["xmin"]),
                int(row["ymin"]),
                int(row["xmax"]),
                int(row["ymax"]),
            )

            # Scale detections to original frame size
            x1 = int(x1 * (original_frame.shape[1] / frame.shape[1]))
            y1 = int(y1 * (original_frame.shape[0] / frame.shape[0]))
            x2 = int(x2 * (original_frame.shape[1] / frame.shape[1]))
            y2 = int(y2 * (original_frame.shape[0] / frame.shape[0]))

            # Render annotations
            color = tuple(class_conf.get("color", DEFAULT_MIDI["color"]))
            if args.debug:
                print(f"Color: {color}")
            cv2.rectangle(original_frame, (x1, y1), (x2, y2), color, thickness=2)
            cv2.putText(
                original_frame,
                class_name,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
            )

            current_detections[class_name] += 1

        # After the number of detections for a class is known, play the corresponding MIDI notes

        if args.debug:
            print(f"Current detections: {current_detections}")

        notes_to_play = defaultdict(dict)

        # Set the MIDI notes to play
        # More detections means more notes
        for class_name, count in current_detections.items():
            for _ in range(count):
                blacklist = set(notes_to_play)
                for note in playing_notes[class_name].items():
                    blacklist.add(note)
                note_list = get_note_list(class_name)
                available = set(note_list) - blacklist

                # If the number of detections eceeds the number of available notes,
                # play another note in the list with a higher velocity
                if not available:
                    mult = count // len(note_list)
                    note = np.random.choice(note_list)
                    notes_to_play[class_name][note] = ()
                    VELOCITY_BASE + VELOCITY_ADD * mult
                else:
                    note = np.random.choice(list(available))
                    notes_to_play[class_name][note] = VELOCITY_BASE

        if args.debug:
            print(f"Notes wanting to play: {notes_to_play}")

        # Turn off notes that are no longer active for a class
        for class_name, notes in playing_notes.items():
            for note, vel in notes.items():
                if note not in notes_to_play[class_name]:
                    midi.send_note_off(note)
                    if args.debug:
                        print(f"note_off: {class_name} -> MIDI {note}")

        # Turn on notes
        for class_name, notes in notes_to_play.items():
            for note, vel in notes.items():
                midi.send_note_on(note, vel)
                if args.debug:
                    print(f"note_on: {class_name} -> MIDI {note}, vel {VELOCITY_BASE}")

        playing_notes = notes_to_play

        if args.debug:
            print(f"Notes playing: {midi.active_notes}")

        cv2.imshow("YOLOv5 Detections", original_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    midi.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
