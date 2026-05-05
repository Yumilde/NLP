# NLP Vibe Coding Showcase (A1-A9)

《自然语言处理》课程 Vibe Coding 作业统一展示仓库。

## 在线展示结构
- Home: 课程作业总览
- A1: 中文词法分析（已由 Flask 迁移为 Streamlit）
- A2-A9: 各次作业模块化页面

## 本地运行
```bash
conda activate py310
pip install -r requirements.txt
streamlit run Home.py
```

## 部署到 Streamlit Community Cloud
1. 推送本仓库到 GitHub
2. 在 Streamlit Cloud 新建应用，选择该仓库
3. Main file path 设为 `Home.py`
4. Python version 选 `3.10`

## 目录说明
- `Home.py`: 总入口
- `pages/`: A1-A9 页面
- `A1`~`A9`: 原始作业材料（报告/Notebook/HTML/核心代码）

## 注意事项
- 首次加载涉及 Hugging Face 模型的页面会较慢
- 受网络环境影响，模型下载可能失败，建议课前至少预热一次
