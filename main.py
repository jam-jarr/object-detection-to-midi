import torch
import cv2
import numpy as np
import mido
import argparse
import json

DEFAULT_MIDI_NOTE = 60
VELOCITY_BASE = 80
MIDI_CHANNEL = 0


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
    type=bool,
    default=False,
    help="Debug mode",
)


class args:
    pass


parser.parse_args(namespace=args)

if args.debug:
    print("Arguments and defaults:")
    print(vars(args))
    print("-" * 50)

# COCO class to MIDI note mapping
with open("class_mapping.json") as f:
    CLASS_TO_MIDI = json.load(f)

if args.debug:
    print(CLASS_TO_MIDI)
    print(CLASS_TO_MIDI.get("person"))
    exit()


# Load model
model = torch.hub.load("ultralytics/yolov5", args.model, pretrained=True)
model.conf = args.conf
model.iou = args.iou

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize MIDI
midi = MidiController("YOLOv5 Detection")
midi.open_port(args.port)

print("Webcam opened. Press 'q' to quit.")

active_detections = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        results = model(frame)
        annotated_frame = np.squeeze(results.render())
        cv2.imshow("Surveillance to MIDI - Webcam", annotated_frame)

        detections = results.pandas().xyxy[0]
        if not detections.empty:
            current_detections = set()

            for _, row in detections.iterrows():
                class_name = row["name"]
                confidence = row["confidence"]
                current_detections.add(class_name)

                if class_name not in active_detections:
                    midi_note = CLASS_TO_MIDI.get(class_name, DEFAULT_MIDI_NOTE)
                    midi.send_note_on(midi_note, VELOCITY_BASE)
                    midi.active_notes[class_name] = midi_note
                    print(
                        f"note_on: {class_name} -> MIDI {midi_note}, vel {VELOCITY_BASE}"
                    )

                active_detections.add(class_name)

            removed = active_detections - current_detections
            for class_name in removed:
                midi_note = CLASS_TO_MIDI.get(class_name, DEFAULT_MIDI_NOTE)
                midi.send_note_off(midi_note)
                if class_name in midi.active_notes:
                    del midi.active_notes[class_name]
                print(f"note_off: {class_name} -> MIDI {midi_note}")
                active_detections.discard(class_name)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    midi.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam released and windows closed.")
