import torch
import numpy
import torchvision
import application

path = '../Pic/material/storage/lipread_pt/MESSAGE/test/MESSAGE_00042.pt'
sequence = torch.load(path)

version, archive = 'aegithal-v1.0.0', 'getReconstruction.onnx'
getReconstruction = application.Service(version, archive)
getReconstruction.loadSession()

group = []
for _, frame in enumerate(sequence):
    assert isinstance(frame, torch.Tensor)
    compression = frame.numpy()[None, :, :, :]
    request = {"compression": compression}
    response = getReconstruction(request)
    reconstruction = response[0]

    group += [reconstruction]
    continue

together = numpy.concatenate(group, axis=0)
torchvision.utils.save_image(
    torch.from_numpy(together),
    'result.jpg',
    value_range=(-1, 1), 
    normalize=True
)


head = 0
tail = 28
z_a = sequence[head]  # (C, H, W)
z_b = sequence[tail]  # (C, H, W)

# 線性插值產生中間 frames
steps = 5
frames = []
for i in range(steps):
    alpha = i / (steps - 1)
    z_t = (1 - alpha) * z_a + alpha * z_b
    compression = z_t.numpy()[None, :, :, :]
    request = {"compression": compression}
    response = getReconstruction(request)
    reconstruction = response[0]
    frames += [reconstruction]
    continue
frames = numpy.concatenate(frames, axis=0)
# 存成圖片 grid 預覽
grid = torchvision.utils.make_grid(torch.from_numpy(frames), nrow=10, padding=2, normalize=True, value_range=(-1, 1))
torchvision.utils.save_image(grid, 'interpolation_grid.png')
# print(f'已儲存 interpolation_grid.png ({steps} frames)')

# 存成影片
# video = (torch.from_numpy(frames) * 0.5) + 0.5  # (T, C, H, W)
# video = (video * 255).to(torch.uint8)
# video = video.permute(0, 2, 3, 1)  # (T, H, W, C)
# torchvision.io.write_video(
#     'interpolation.mp4',
#     video,
#     fps=15
# )
# print('已儲存 interpolation.mp4')
