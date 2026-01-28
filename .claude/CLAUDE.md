# Ae 專案規則

本專案的開發規範分為以下幾個部分，執行相關任務前請先閱讀對應的規則檔案。

## 規則檔案

| 檔案 | 說明 |
|------|------|
| `.claude/code-style.md` | 程式碼風格規範 |
| `.claude/git-rules.md` | Git commit 與分支規範 |
| `.claude/api-design.md` | API 與模組設計規範 |
| `.claude/spec.md` | 需求提交範本 |

## 快速參考

### 程式碼風格重點

- Class: PascalCase (`Model`, `Framework`)
- Method: camelCase + 動詞開頭 (`getCompression`, `loadWeight`)
- return 使用括號: `return(value)`
- 條件式使用括號: `if(condition):`
- 迴圈結尾明確 `continue`
- Class 結尾使用 `pass`

詳細規則請參閱 `.claude/code-style.md`
