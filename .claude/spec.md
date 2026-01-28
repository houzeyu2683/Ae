# 需求提交範本

向 AI agent 提交需求時可參考以下格式。

---

## 1. 簡單版

適用於小功能、單一方法。

```
在 sparrow 新增一個方法，可以計算 loss 的平均值
```

```
幫我在 Dashboard 加一個方法，可以記錄 learning rate
```

---

## 2. 標準版

適用於需要明確輸入輸出的功能。

```
功能：計算多個 loss 的加權平均
位置：sparrow/_model_.py 的 Model class
輸入：losses (dict), weights (dict)
輸出：加權平均值 (torch.Tensor)
```

---

## 3. 完整版

適用於複雜功能、多個步驟。

```
功能：新增資料擴增功能
位置：material/_unit_.py

需求：
- 隨機水平翻轉
- 隨機旋轉 -15 到 15 度
- 機率各 50%

輸入：image (torch.Tensor)
輸出：augmented image (torch.Tensor)

備註：只在訓練時使用
```

---

## 提示

- 簡單功能一句話即可
- 複雜功能多說明細節
- 可指定位置（檔案、class）或讓 agent 決定
- 可附上參考程式碼或連結
