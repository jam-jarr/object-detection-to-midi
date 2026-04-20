import torch
import cv2
import numpy as np
import mido

# COCO class to MIDI note mapping
CLASS_TO_MIDI = {
    "person": 60,
    "bicycle": 62,
    "car": 64,
    "motorcycle": 65,
    "airplane": 67,
    "bus": 69,
    "train": 71,
    "truck": 72,
    "boat": 74,
    "traffic light": 76,
    "fire hydrant": 77,
    "stop sign": 79,
    "parking meter": 81,
    "bench": 83,
    "bird": 48,
    "cat": 50,
    "dog": 52,
    "horse": 53,
    "sheep": 55,
    "cow": 57,
    "elephant": 59,
    "bear": 36,
    "zebra": 38,
    "giraffe": 40,
    "backpack": 42,
    "umbrella": 43,
    "handbag": 45,
    "tie": 47,
    "suitcase": 84,
    "frisbee": 86,
    "skis": 88,
    "snowboard": 89,
    "sports ball": 91,
    "kite": 93,
    "baseball bat": 95,
    "baseball glove": 96,
    "skateboard": 98,
    "surfboard": 100,
    "tennis racket": 102,
    "bottle": 104,
    "wine glass": 106,
    "cup": 108,
    "fork": 110,
    "knife": 112,
    "spoon": 114,
    "bowl": 116,
    "banana": 117,
    "apple": 119,
    "sandwich": 121,
    "orange": 122,
    "broccoli": 124,
    "carrot": 126,
    "chair": 24,
    "couch": 26,
    "potted plant": 28,
    "bed": 30,
    "dining table": 32,
    "toilet": 34,
    "tv": 12,
    "laptop": 14,
    "mouse": 16,
    "keyboard": 17,
    "remote": 19,
    "microwave": 21,
    "oven": 23,
    "toaster": 0,
    "refrigerator": 2,
    "book": 4,
    "clock": 5,
    "vase": 7,
    "scissors": 9,
    "teddy bear": 11,
}

DEFAULT_MIDI_NOTE = 60
VELOCITY_BASE = 80
VELOCITY_SCALE = 20
MIDI_CHANNEL = 0


class MidiController:
    def __init__(self, port_name="YOLOv5 Detection"):
        self.port = None
        self.port_name = port_name
        self.use_file_mode = False
        self.midi_file = None
        self.track = None
        self.active_notes = {}

    def open_port(self):
        available_ports = mido.get_output_names()
        print(f"Available MIDI outputs: {available_ports}")

        try:
            self.port = mido.open_output(self.port_name, virtual=True)
            print(f"Virtual MIDI port created: {self.port_name}")
            return True
        except Exception as e:
            print(f"Could not create virtual port: {e}")

        if available_ports:
            try:
                self.port = mido.open_output(available_ports[0])
                print(f"Connected to MIDI port: {available_ports[0]}")
                return True
            except Exception as e:
                print(f"Could not open port: {e}")

        print("No MIDI ports available. Using file mode.")
        self.use_file_mode = True
        self.midi_file = mido.MidiFile()
        self.track = mido.MidiTrack()
        self.midi_file.tracks.append(self.track)
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
        for class_name, note in list(self.active_notes.items()):
            self.send_note_off(note)

        if self.use_file_mode and self.midi_file:
            self.track.append(mido.Message("end_of_track"))
            self.midi_file.save("detections.mid")
            print("MIDI file saved: detections.mid")
        elif self.port:
            self.port.close()
            print("MIDI port closed")


# Load model
model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
model.conf = 0.30
model.iou = 0.45

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Initialize MIDI
midi = MidiController("YOLOv5 Detection")
midi.open_port()

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
