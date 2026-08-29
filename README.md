# Capital Bikeshare DC 数据分析项目

本项目基于 Capital Bikeshare 月度骑行 CSV，已经实现以下功能：

- 数据读取、清洗和特征工程；
- 时间规律与用户类型分析；
- 站点、OD 和 500 米网格空间分析；
- Top 10 网格下一小时出发需求预测；
- Streamlit 交互式可视化仪表板；
- 基于 GBFS 实时快照的网格级动态调度策略模拟（选做部分）。

## 目录与数据格式

原始月度文件需保持以下结构，月份使用 `YYYYMM`：

```text
data/
└── 202501-capitalbikeshare-tripdata/
    └── 202501-capitalbikeshare-tripdata.csv
```

程序只读取符合该命名规则的文件，会忽略 `__MACOSX`、临时文件和 `data/processed/` 中的处理结果。

主要入口：

- `src/preprocess.py`：清洗与特征工程；
- `src/temporal.py`：时间和用户类型分析；
- `src/spatial_od.py`：空间与 OD 分析；
- `src/grid.py`：空间分析、预测和调度共用的 500 米网格映射；
- `src/forecast.py`：网格级需求预测；
- `src/realtime_gbfs.py`：获取、校验并保存一次 GBFS 站点库存快照；
- `src/rebalancing.py`：构建历史流量画像并比较调度策略；
- `src/dashboard_data.py`：生成仪表板聚合数据；
- `dashboard.py`：Streamlit 仪表板。

## 安装依赖

建议在虚拟环境中运行：

```powershell
python -m pip install -r requirements.txt
```

## 运行流程

### 1. 数据预处理

完整项目建议一次性处理 2025 年分析/训练数据和 2026 年 1—7 月测试数据，全部写入统一目录 `data/processed/`：

```powershell
python -m src.preprocess `
  --start-month 202501 `
  --end-month 202607 `
  --start-time 2025-01-01 `
  --end-time 2026-08-01
```

默认输出 Parquet 和质量报告到 `data/processed/`。时间、空间、预测和仪表板模块均从该目录按月份选择文件。程序默认不覆盖已有文件，确认重跑时添加 `--overwrite`。

可先执行不写文件的冒烟测试：

```powershell
python -m src.preprocess `
  --start-month 202601 `
  --end-month 202601 `
  --max-rows 10000 `
  --dry-run
```

### 2. 时间分析

```powershell
python -m src.temporal `
  --start-month 202501 `
  --end-month 202512 `
  --start-time 2025-01-01 `
  --end-time 2026-01-01 `
  --frequency day `
  --overwrite
```

`--frequency` 支持 `hour`、`day`、`week` 和 `month`。总骑行距离趋势固定输出为折线图；其余结果包括24小时平均骑行量柱状图、工作日/周末对比和用户类型分析，统一写入 `outputs/temporal/`。

### 3. 空间与 OD 分析

```powershell
python -m src.spatial_od `
  --start-month 202501 `
  --end-month 202512 `
  --grid-size-m 500 `
  --overwrite
```

结果写入 `outputs/spatial_od/`，包括站点排名、Top 20 OD、500 米网格统计、早晚高峰净流入，以及站点热力图、网格地图和 OD Sankey。

### 4. 出行需求预测

```powershell
python -m src.forecast --overwrite
```

默认使用 2025 年数据选择出发量最高的 10 个网格并训练模型，对 2026 年 1—7 月进行时间外测试。结果写入 `outputs/forecasting/`，包括：

- 周周期朴素基线、历史分组均值和 HGB 模型的评估结果；
- 逐小时预测、总体和逐月指标；
- Top 10 网格、特征结构、模型文件和诊断图。

如不需要 2026 年测试，可使用 `--skip-test`。测试结束月份可通过 `--test-end-month YYYYMM` 调整。

### 5. 交互式仪表板

仪表板依赖第 3 步生成的站点/OD 表和第 4 步生成的预测结果。先生成轻量聚合数据：

```powershell
python -m src.dashboard_data --overwrite
```

再启动应用：

```powershell
python -m streamlit run dashboard.py
```

浏览器访问 `http://localhost:8501`。仪表板包括：

- 2025 年日期和用户类型筛选；
- 骑行量、时长、距离和 24 小时活动分布；
- 站点与网格需求地图；
- Top 10 节点间双列 OD Sankey（起终点分离，并按纬度由北到南排列）；
- 2026 年 1—7 月模型、网格和日期范围可切换的预测评估；
- GBFS 快照库存地图、调度参数交互和无调度/贪心调度对比。

### 6. 实时快照与动态调度模拟

该部分使用 Capital Bikeshare 的公开 GBFS 数据。一次快照只表示获取时刻各站点的可用车辆和空桩，不包含持续变化轨迹；因此代码将快照作为模拟初始库存，并用 2025 年 Top 10 网格的“星期几 × 小时”平均出发量和到达量表示未来短时需求。

先构建历史网格流量画像（完成第 4 步后只需执行一次）：

```powershell
python -m src.rebalancing prepare-profiles --overwrite
```

获取并保存最新快照：

```powershell
python -m src.realtime_gbfs --overwrite
```

运行默认的 6 小时反事实模拟：

```powershell
python -m src.rebalancing simulate --overwrite
```

默认策略在每个模拟小时开始前，根据预测出发/到达量识别低库存网格和高库存网格，再按网格间距离从近到远调车。策略同时受车辆数、单车载量、每小时最大搬运量、站点容量及 25%/75% 库存率阈值约束。常用参数示例：

```powershell
python -m src.rebalancing simulate `
  --horizon-hours 8 `
  --demand-multiplier 1.5 `
  --truck-count 4 `
  --truck-capacity 12 `
  --max-relocated-bikes-per-hour 48 `
  --overwrite
```

模拟同时运行“无调度”和“贪心调度”两种情景，并比较未满足出发、失败还车、空/满网格小时、库存不平衡 bike-hours、搬运车辆数和搬运 bike-km。结果写入 `outputs/rebalancing/`，仪表板的 **Live rebalancing** 页签也可以刷新快照并交互调整主要参数。

还可将 2026 年已有预测与实际出发量作为历史需求重放：

```powershell
python -m src.rebalancing simulate `
  --mode backtest `
  --start-time "2026-07-01 08:00" `
  --horizon-hours 6 `
  --overwrite
```

需要注意：当前没有历史 GBFS 库存档案，`backtest` 模式仍以选定的一次快照作为初始库存，到达量仍使用历史画像，因此属于历史需求重放，不是对运营方真实调度的完整回测。若要严格评估真实效果，需要定时采集快照，或获得历史站点库存、实际调度记录及车辆轨迹。

## 当前清洗与建模口径

- 删除 `ride_id`、起止时间或用户类型无效的记录，并按 `ride_id` 跨月份去重；
- 骑行时长由起止时间计算，仅保留 60 秒至 4 小时；
- 经纬度按 D.C. 都会区范围校验，异常坐标置为空，但不会因此删除仍可用于时间分析的行程；
- `distance_km` 为 Haversine 起终点直线位移，不是道路实际骑行里程；
- 工作日为周一至周五且排除美国联邦节假日；
- 交通时段为早高峰 07:00—09:59、晚高峰 16:00—18:59、夜间 22:00—05:59，其余为平峰；
- 预测特征仅使用日历、历史滞后和已 `shift(1)` 的滚动统计；
- 调度模拟只覆盖预测模型的 Top 10 网格，GBFS 快照按同一 500 米网格口径聚合；
- 一次实时快照不会被当作真实需求序列，短时出发/到达量来自历史画像；
- 原始数据没有单个用户 ID，因此用户“使用频率”仅表示各用户类型的行程次数和占比。

## 输出目录

```text
data/processed/          清洗后的月度 Parquet 和质量报告
data/realtime/           带 UTC 时间戳的 GBFS 站点库存快照与元数据
outputs/temporal/        时间分析图表与统计表
outputs/spatial_od/      空间、网格、OD 图表与统计表
outputs/forecasting/     预测结果、指标、模型与诊断图
outputs/rebalancing/     调度轨迹、指令、策略指标、对比图和库存地图
outputs/dashboard/data/  仪表板聚合 Parquet
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖预处理、跨块去重、时间聚合、站点与网格统计、预测特征防泄漏、仪表板聚合一致性、GBFS 快照标准化、统一网格映射、历史流量画像和调度约束。

## License

MIT