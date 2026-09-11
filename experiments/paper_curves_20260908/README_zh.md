# 当前论文图与原始曲线

[按问题浏览](../../figures/paper_curves_20260908/index.html) · [正文PDF](../../figures/paper_curves_20260908/main_figures.pdf) · [W&B raw曲线](../../figures/paper_curves_20260908/raw_curves.html)

30张图；正文9张；10条训练记录；52组核验评估；256,276个浏览器原始点。

配置与两篇参考论文的对应关系见[设计说明](DESIGN_zh.md)。英文[caption](FIGURE_CAPTIONS.md)与[数据验证](validation_report.json)一起提供。

重画：`python experiments/plot_paper_figures.py`。所有原始数据已冻结，无需连接原训练目录。

数据与图核验：`python experiments/validate_paper_figures.py`。浏览器检查记录见`browser_validation.json`。
