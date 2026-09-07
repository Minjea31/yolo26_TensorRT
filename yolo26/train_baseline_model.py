from ultralytics import YOLO
import torch
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='baseline')
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--model_pt', type=str, default='../yolo26n.pt')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--data', type=str, default="../dataset.yaml")
parser.add_argument("--device", type=str, default='0')

args = parser.parse_args()

project_dir = Path(__file__).resolve().parent / "checkpoints"
model = YOLO(args.model_pt)
if not args.resume:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=4, save_period=5, project=str(project_dir))
else:
    model.train(data=args.data, epochs=args.epoch, imgsz=640, device=args.device, name=args.name,
                batch=args.bs, workers=4, save_period=5, project=str(project_dir), resume=args.resume)
