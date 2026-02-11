import lani

device = 'cpu'
model = lani.Model(device)
model.activateLayer()

history = './log/lani-2026-0203/'
framework = lani.Framework(model, device, history)

# # aggregate = framework.Aggregate(history)
checkpoint = [
    '250000.pt',
    '260000.pt',
    '270000.pt',
]
framework.saveWeight(checkpoint)


# import torch
# import glob

# # 取得所有 checkpoint 檔案
# checkpoint_paths = sorted(glob.glob('./log/laniidae-2026-0203/weight/*.pt'))
# # 或手動指定
# # checkpoint_paths = ['1000.pt', '1001.pt', '1002.pt']

# # 載入第一個 checkpoint 作為基底
# avg_state_dict = torch.load(checkpoint_paths[0], map_location='cpu')

# # 累加其他 checkpoint 的權重
# for path in checkpoint_paths[1:]:
#     state_dict = torch.load(path, map_location='cpu')
#     for key in avg_state_dict:
#         avg_state_dict[key] += state_dict[key]

# # 除以 checkpoint 數量取平均
# n = len(checkpoint_paths)
# for key in avg_state_dict:
#     avg_state_dict[key] /= n

# # 儲存平均後的權重
# torch.save(avg_state_dict, './averaged_weight.pt')

# # 載入到模型
# model.load_state_dict(avg_state_dict)