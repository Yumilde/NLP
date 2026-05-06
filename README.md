# NLP 课程 A1-A9 Vibe Coding Showcase

## 目录结构
- `app.py`: 总入口主页
- `pages/`: `A1-A9` 分页展示
- `shared/adapter.py`: 复用原作业代码的适配层
- `A1-A9/`: 原始作业文件

## 本地运行
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 部署
1. 推送仓库到 GitHub（公开或可访问）。
2. 打开 Streamlit Cloud，新建 App。
3. 选择仓库与分支，Main file path 填 `app.py`。
4. 等待自动安装 `requirements.txt` 并启动。

## 课堂演示顺序（10-15 分钟）
1. `A1`：词法分析开场（分词和词性）
2. `A2-A4`：句法到语义（结构理解）
3. `A5-A7`：篇章与信息抽取（高阶任务）
4. `A8-A9`：翻译与情感（应用落地）

## 冷启动与降级预案
- 首次加载较慢的模块：`A5/A6/A8/A9`（含 `transformers/torch`）。
- 建议课前在云端依次点击一次每个页面预热缓存。
- 若现场网络不稳：使用页面默认输入和预置示例优先演示结果。

