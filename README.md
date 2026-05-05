# NLP Vibe Coding Showcase (A1-A9)

《自然语言处理》课程 Vibe Coding 作业统一展示仓库。

## 在线演示
https://nlpvibecoding.streamlit.app/

## 在线展示结构
- Home: 课程作业总览
- A1-A9: 各次作业模块化页面

## 本地运行
```bash
conda activate py310
pip install -r requirements.txt
streamlit run Home.py
```

## 目录说明
- `Home.py`: 总入口
- `pages/`: A1-A9 页面
- `A1`~`A9`: 原始作业材料（报告/Notebook/HTML/核心代码）

## A8 例句
- Machine translation has evolved rapidly in recent years.
- It rains cats and dogs.
- 下起了倾盆大雨。

## A9 例句
- 这款耳机音质非常好，续航也很稳定，性价比很高。
- 这款相机成像效果太棒了，我非常满意。
- 这台手机玩游戏半小时就没电了。

## 注意事项
- 首次加载涉及 Hugging Face 模型的页面会较慢
- 受网络环境影响，模型下载可能失败，建议课前至少预热一次
