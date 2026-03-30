from ultralytics import YOLO


def put_in_eval_mode(trainer):
    n_layers = trainer.args.freeze
    if not isinstance(n_layers, int):
        return
    for i, (name, module) in enumerate(trainer.model.named_modules()):
        if name.endswith("bn") and int(name.split(".")[1]) < n_layers:
            module.eval()
            module.track_running_stats = False
            # print(f"{name} set eval")


model = YOLO("./weights/yolo26n.pt")  # load a pretrained model (recommended for training)
model.add_callback("on_train_start", put_in_eval_mode)
model.train(data=r"/data.yaml", epochs=500, imgsz=640, device="cuda",amp=False,batch=8,freeze=23)# freeze=23 for yolo26