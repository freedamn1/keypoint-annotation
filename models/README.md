# Trained model artifacts

- `pose-best.pt`: YOLO11 pose model trained for 150 epochs, from `maize-yolo11s-pose-2`.
- `ear-best.pt`: YOLO11 ear detector trained for 150 epochs, from `maize-yolo11s-ear-2`.
- `metadata.json`: the 85% center crop ratio and ordered keypoint labels required by `infer_labelme.py`.

These are the trained checkpoints selected by validation, not the original `yolo11s*.pt` download files. The iPhone17PM predictions previously made with these models contained incorrect keypoints on many images; review predictions before using them as labels.

SHA-256:

```text
pose-best.pt  115c7a17e4cfd0660d70fa2fce76d60d1e90109e1ddc85ed741b9083d8641fdf
ear-best.pt   cf912ef85bbf6d5914d5aed0e58a3bb6a0aadabd0fe280a818ed6713ab850370
```
