import torch
import cv2
import numpy as np
import mido
import argparse
import json
from ultralytics import YOLO

VELOCITY_BASE = 80
MIDI_CHANNEL = 0

DEFAULT_MIDI = {
    "note": 60,
    "color": (0, 255, 0),
}


class MidiController:
    def __init__(self, port_name="YOLOv5 Detection"):
        self.port = None
        self.port_name = port_name
        self.use_file_mode = False
        self.midi_file = None
        self.track = None
        self.active_notes = {}

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

    def close(self):
        for _class_name, note in list(self.active_notes.items()):
            self.send_note_off(note)

        if self.use_file_mode and self.midi_file:
            if self.track:
                self.track.append(mido.Message("end_of_track"))
                self.midi_file.save("detections.mid")
            print("MIDI file saved: detections.mid")
        elif self.port:
            self.port.close()
            print("MIDI port closed")


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

# COCO class to MIDI note mapping
with open("configuration.json") as f:
    configuration = json.load(f)

if args.debug:
    print(configuration)
    print(configuration.get("person"))


# Load model
# model = torch.hub.load(
#     "ultralytics/yolov5", args.model, pretrained=True, _verbose=False
# )

model = YOLO("yolov8s.pt")

model.conf = args.conf
model.iou = args.iou

model.cuda()

# Initialize webcam
cap = cv2.VideoCapture(args.device)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize MIDI
midi = MidiController("YOLOv5 Detection")
midi.open_port(args.port)

print("Webcam opened. Press 'q' to quit.")

active_detections = set()

cv2.namedWindow("YOLOv5 Detections", cv2.WINDOW_NORMAL)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
        print(results.pandas().xyxy[0])
        if not detections.empty:
            current_detections = set()

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

                current_detections.add(class_name)
                if class_name not in active_detections:
                    midi_note = class_conf.get("note")
                    midi.send_note_on(midi_note, VELOCITY_BASE)
                    midi.active_notes[class_name] = midi_note
                    print(
                        f"note_on: {class_name} -> MIDI {midi_note}, vel {VELOCITY_BASE}"
                    )

                active_detections.add(class_name)

            removed = active_detections - current_detections
            for class_name in removed:
                midi_note = class_conf.get("note")
                midi.send_note_off(midi_note)
                if class_name in midi.active_notes:
                    del midi.active_notes[class_name]
                print(f"note_off: {class_name} -> MIDI {midi_note}")
                active_detections.discard(class_name)

        cv2.imshow("YOLOv5 Detections", original_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    midi.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
