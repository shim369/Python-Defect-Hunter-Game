# 画面遷移図: Python-Defect-Hunter-Game

## 画面一覧
| 画面 | 内容 |
|---|---|
| タイトル | ゲーム説明、スタートボタン |
| 出題 | 画像表示、Canvas でなぞる、「欠陥なし」「提出」「やり直し」、ブラシの太さ、進捗バー |
| 結果(1 問) | 元画像・正解・あなた・AI 予測を並べて表示。IoU の点数と勝敗(AI は初期しきい値 0.5 で固定) |
| 成績 | 5 問終了時は「中間成績」、10 問終了時は「最終結果」(同じ画面で見出しだけ変わる)。あなたと AI の Precision / Recall / F1 / 平均 IoU / TP・FP・FN・TN。**AI のしきい値スライダー**はここにある |

全画面共通: 右上に **日本語 / EN スイッチ**(選択はブラウザに保存)。URL に `?seed=整数` を付けると出題が固定される。

## 遷移図
```mermaid
stateDiagram-v2
    [*] --> Title
    Title --> Question: スタート (10 問を出題)

    state Question {
        [*] --> Drawing
        Drawing --> Drawing: なぞる / やり直し
        Drawing --> Submitted: 提出 (何も描かなければ「欠陥なし」扱い)
        Drawing --> Submitted: 欠陥なし
    }

    Question --> Result: 採点 (IoU)
    Result --> Question: 次へ (5 問・10 問目以外)
    Result --> Stats: 成績を見る (5 問目・10 問目)
    Stats --> Stats: しきい値スライダー変更 (AI の指標を再計算)
    Stats --> Question: 続ける (5 問目の後)
    Stats --> Title: もう一度遊ぶ (10 問目の後)
```

## データフロー
```mermaid
sequenceDiagram
    participant B as ブラウザ (HTML/Canvas/JS)
    participant S as FastAPI (出題・採点・指標。モデルは動かさない)
    participant D as 事前生成データ (画像, 正解マスク, AI の確率マップ PNG)

    B->>S: GET /api/config (画像サイズ, 初期しきい値)
    B->>S: GET /api/round?seed= (出題リスト)
    S->>D: 欠陥あり 6 : 良品 4 で画像 ID を選ぶ
    S-->>B: 画像 URL, ID (欠陥の有無は返さない)
    B->>B: プレイヤーがなぞる
    B->>S: POST /api/answer (ID, なぞったマスク or 欠陥なし)
    S->>D: 正解マスク, AI の確率マップを読込
    S->>S: IoU 計算 (自前指標), AI はしきい値 0.5 で 2 値化
    S-->>B: 点数, 勝敗に必要な値, 正解/確率マップの URL
    B->>B: 4 枚のパネルを描画 (確率マップは初期しきい値で色付け)
    B->>S: POST /api/summary (回答済みの ID と「欠陥ありと答えたか」, しきい値)
    S->>S: 正解はサーバー側の値を使い、あなたと AI の P/R/F1 を計算
    S-->>B: 成績表 (スライダーを動かすたびに再計算)
```
