# Code Style Guide

這份文件定義了本專案的程式碼風格規範，供 AI agent 撰寫程式碼時遵循。

---

## 1. 命名規範

### 1.1 檔案與資料夾

| 類型 | 規則 | 範例 |
|------|------|------|
| 資料夾 | 單一單字，全小寫 | `material`, `sparrow`, `robin` |
| 模組檔案 | 單一單字，前後加底線 `_name_.py` | `_hub_.py`, `_model_.py` |
| 腳本檔案 | 連字號分隔 `script-module-action.py` | `script-sparrow-fit.py` |
| 初始化 | `__init__.py` (Python 原生慣例) | |

**規則: 資料夾與檔案不可同名**

```
# 正確
material/_hub_.py
sparrow/_model_.py

# 錯誤
data/_data_.py
model/_model_.py
```

### 1.2 程式碼元素

| 類型 | 規則 | 範例 |
|------|------|------|
| Class | 單一名詞，首字母大寫 | `Model`, `Framework`, `Hub` |
| Function / Method | 動詞+名詞，動詞小寫，名詞大寫 | `getCompression`, `loadWeight` |
| Variable | 單一單字，小寫，優先名詞 | `device`, `model`, `batch` |
| Constant | 單一單字，小寫 | `rate`, `size` |

---

## 2. 函數/方法規範

### 2.1 命名動詞

**規則: 有 return 值必須使用 `get` 開頭**

| 動詞 | 用途 | 回傳值 | 範例 |
|------|------|--------|------|
| `get` | 取得/計算資料 | 有 | `getCompression`, `getCriteria` |
| `load` | 載入外部資源 | bool | `loadWeight`, `loadVersion` |
| `save` | 儲存資料 | bool | `saveWeight`, `saveComparison` |
| `make` | 執行並存至 self | bool | `makeComparison`, `makeInference` |
| `activate` | 初始化/啟用 | bool | `activateLayer` |
| `open` / `close` | 開啟/關閉資源 | bool | `openSession`, `closeSession` |
| `insert` | 插入資料 | bool | `insertStatistic`, `insertPicture` |
| `fit` | 訓練相關 | bool | `fitWeight` |

### 2.2 參數命名

**規則: 參數名稱不可包含函數/方法名稱中的名詞**

| 函數/方法 | 正確參數 | 錯誤參數 |
|-----------|----------|----------|
| `getValue` | `source`, `data` | `value` |
| `loadWeight` | `path` | `weight` |
| `saveComparison` | `archive`, `path` | `comparison` |
| `getReconstruction` | `compression` | `reconstruction` |

### 2.3 布林回傳

執行動作的函數/方法回傳 `bool`，成功時回傳 `True`:

```python
def saveWeight(self, path: str) -> bool:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    safetensors.torch.save_file(self.model.state_dict(), path)
    return(True)
```

### 2.4 Type Hints

在函數/方法簽名中使用 type hints:

```python
def __init__(self, device: str) -> None:
def loadWeight(self, path: str) -> bool:
def getCompression(self, image: torch.Tensor) -> torch.Tensor:
```

---

## 3. 語法風格

### 3.1 縮排區塊結尾

**規則: 所有縮排區塊必須明確標記結束**

| 區塊類型 | 結尾關鍵字 |
|----------|------------|
| `class` | `pass` |
| `def` | `return` 或 `return(value)` |
| `if` / `elif` / `else` | `pass` |
| `with` | `pass` |
| `for` | `continue` 或 `break` |
| `while` | `continue` 或 `break` |
| `try` / `except` / `finally` | `pass` |

### 3.2 避免連續縮排

**規則: 避免巢狀結構，不可連續縮排**

```python
# 錯誤
if(condition):
    try:
        processData()
        pass
    pass

# 錯誤
for item in iteration:
    if(valid):
        handleItem(item)
        pass
    continue

# 正確
validated = getValidation(condition)
if(validated):
    processData()
    pass
```

### 3.3 括號使用

**Return 語句:**

```python
return(length)
return(True)
```

**條件語句:**

```python
if(not here):
    pass

if(number%accumulation==0):
    pass

while(True):
    pass
```

### 3.4 迴圈結構

迴圈結尾使用 `continue`，迴圈結束後指派 `_ = iteration`:

```python
iteration = paper.readlines()
for item in iteration:
    path = item.replace("\n", "")
    queue += [os.path.join(folder, path)]
    continue
_ = iteration
```

---

## 4. Class 結構

### 4.1 範本

```python
class Class:

    def __init__(self) -> None:
        return

    pass
```

**重點:**
- `class` 後空一行
- `__init__` 結尾使用 `return`
- `__init__` 後空一行
- `class` 結尾使用 `pass`

### 4.2 方法別名

使用賦值方式定義別名:

```python
class Model(torch.nn.Module):
    # ...
    forward = getCriteria
    pass

class Unit(torch.utils.data.Dataset):
    # ...
    __len__ = getLength
    __getitem__ = getItem
    pass
```

---

## 5. 其他慣例

### 5.1 Import 順序

1. 標準庫
2. 第三方套件
3. 本地模組

每個 import 獨立一行:

```python
import os
import torch
import safetensors.torch
import tensordict
import visualization
```

### 5.2 路徑處理

使用 `os.path.join()` 組合路徑:

```python
folder = os.path.join(self.history, tag)
checkpoint = os.path.join(self.history, 'weight', f'{number}.pt')
```

### 5.3 字串格式化

使用 f-string:

```python
checkpoint = f'{number}.pt'
archive = f'{index}-sample.jpg'
```

### 5.4 屬性存取

適時使用 `getattr`:

```python
getCompression = getattr(self.model, 'getCompression')
compression = getCompression(image)
```

### 5.5 TensorDict 使用

使用 `tensordict.TensorDict` 封裝多個 tensor:

```python
criteria = tensordict.TensorDict(device=self.device)
criteria.set("pixel", pixel)
criteria.set("total", total)
return(criteria)
```

### 5.6 註解

可使用中文或英文:

```python
# optimization
rate = 1e-4

block_out_channels=[32, 64, 128, 256],  # 多一層 downsampling
```

---

## 6. 完整範例

### 6.1 __init__.py

```python
from ._model_ import *
from ._framework_ import *

__all__ = ['Model', 'Framework']
```

### 6.2 Class 完整範例

```python
import os
import torch
import tensordict

class Example:

    def __init__(self, device: str) -> None:
        self.device = device
        return

    def getData(self, size: int) -> torch.utils.data.DataLoader:
        # implementation
        return(data)

    def processItem(self, item: torch.Tensor) -> bool:
        item = item.to(self.device, non_blocking=True)
        # processing logic
        return(True)

    pass
```
