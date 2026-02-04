相同條件：
資料集：約294萬張影像

在訓練 sparrow ，使用 MSE ，不給約束的單純 auto-encoder 在復原圖片滿不錯的

訓練robin，使用VQ的作法，MSE loss 會在約0.003這邊上下震盪，嘗試用了一些「非改結構」的作法來改善

把 VQ loss term 用一個 decay 0.8 縮小，讓 訓練比較偏向 重構誤差(MSE)
把 gradient accumulation 調高一點，模擬大 batch size
