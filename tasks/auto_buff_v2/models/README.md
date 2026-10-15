# 打符模型

`models/` 负责从检测结果估计状态，以及按该状态预测未来位置。顶层
`buff_frame_processor` 通过 `RuneModel` 接口使用模型，瞄准器通过
`RuneEstimate` 和 `RunePredictor` 使用结果。

- `rune_model.hpp`：公共模型接口、状态快照和重投影结果。
- `rune_model_factory.hpp`：按 `buff_v2.model.type` 创建模型。
- `rune_predictor.*`：两种模型共用的匀速、正弦运动外推与靶心位置计算；不依赖具体模型的滤波状态。
- `njust_rune_model.*`：Njust 五叶重投影 EKF、初始化与观测校正。
- `njust_rune_config.*`：解析 `buff_v2.model.njust`，包括大小符运动拟合配置。
- `njust_ekf_state.hpp`、`njust_energy_fitter.*`：Njust 专属滤波状态与加权运动拟合；拟合参数由拟合器自身的 `Config` 持有。
- `hungarian.hpp`：模型内部的观测分配工具，当前由 Njust 使用。
- `sp_rune_model.*`、`sp_rune_config.*`、`sp_rune_solver.*`、`sp_rune_tracker.*`、`sp_motion_fitter.*`：SP 模型及其专属配置、解算、跟踪与拟合。

`BuffConfig::Model` 只保存模型选择和原始配置节点。具体模型解析自身配置；公共预测器保留两种模型都需要的运动表达，不引用 Njust 或 SP 的内部滤波类型。
