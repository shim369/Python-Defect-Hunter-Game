# 画面遷移図: Python-Defect-Hunter-Game

## 画面一覧
| 画面 | 内容 |
|---|---|
| タイトル | ゲーム説明、スタートボタン |
| 出題 | 画像表示、Canvas でなぞる、「欠陥なし」「提出」「やり直し」 |
| 結果(1 問) | 正解マスク、プレイヤー、AI 予測を並べて表示。IoU 点数比較。しきい値スライダー(任意) |
| 中間成績 | 数問ごとにプレイヤーの Precision / Recall / F1(AI と比較) |
| 最終結果 | 総合成績、AI との勝敗、もう一度遊ぶ |

## 遷移図
```mermaid
stateDiagram-v2
    [*] --> Title
    Title --> Question: スタート

    state Question {
        [*] --> Drawing
        Drawing --> Drawing: なぞる / やり直し
        Drawing --> Submitted: 提出
        Drawing --> Submitted: 欠陥なし
    }

    Question --> Result: 採点 (IoU)
    Result --> Result: しきい値スライダー変更 (任意)
    Result --> Summary: 数問ごと (N 問目)
    Result --> Question: 次の問題
    Result --> Final: 最終問題
    Summary --> Question: 続ける
    Summary --> Final: 最終問題だった場合
    Final --> Title: タイトルへ
    Final --> Question: もう一度遊ぶ
```

## データフロー
```mermaid
sequenceDiagram
    participant B as ブラウザ (HTML/Canvas/JS)
    participant S as FastAPI (静的配信 + 出題 API)
    participant D as 事前生成データ (画像, 正解マスク, 予測PNG)

    B->>S: GET /api/round (出題リスト)
    S->>D: 画像 ID を選択
    S-->>B: 画像 URL, ID
    B->>B: プレイヤーがなぞる
    B->>S: POST /api/answer (ID, 塗り or 欠陥なし)
    S->>D: 正解マスク, AI 予測を読込
    S->>S: IoU 計算 (自前指標)
    S-->>B: 点数, 正解, AI 予測, AI 点数
    B->>B: しきい値スライダーで確率マップを再二値化 (任意)
```
